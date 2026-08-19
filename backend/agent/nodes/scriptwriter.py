from typing import List

from langchain_core.output_parsers import PydanticOutputParser

from backend.agent.state import AgentState, ScriptOutput
from LLM.llm import LLM


SCRIPTWRITER_SYSTEM_PROMPT = """
You are an expert educational scriptwriter specializing in deep-dive, long-form animated mathematical and scientific explainer videos (3-10 minutes total).

## Your task
Given a topic and research notes, write a highly structured, in-depth video script with 5-8 scenes. Do NOT write surface-level summaries; explain the "why" and "how" deeply.

## Scene structure pattern
A concise deep-dive script must follow this 3 to 5-scene arc:
1. **Hook & Intuition** (Scene 1): The real-world problem or paradox to grab attention.
2. **First Principles & Core Logic** (Scene 2, or 2-3): Breaking down the core components, variables, and the essential mathematics or mechanisms behind them.
3. **Application & Example** (Scene 3, or 3-4): Running real numbers, visualizing a concrete example, or tracing the system in action.
4. **Synthesis** (Final Scene): The grand takeaway and conclusion.

## For each scene, you must provide:

**title**: 3-6 words defining the scene's exact focus.

**narration** (2-3 sentences, 20-90 words):
- Spoken by voiceover. Write CONVERSATIONALLY but ACADEMICALLY.
- Explain the mechanics deeply. Don't just state facts; walk the viewer through the logic step-by-step.
- Use analogies for abstract concepts, followed by the rigorous definition.
- Make sure the transitions between scenes are smooth and logical. Each scene should build on the previous one.
- Be careful that the narration have a smooth end.
- DO NOT MAKE IT TOO BIG.

## Output format
JSON object with:
- `title`: string
- `scenes`: array of 5-8 Scene objects (id starts at 1)
- Before writing the scenes, use the thought_process key to calculate your word counts, durations, and scene limits. Then, output the final scenes.

A scene object has:
- `id`: integer (1, 2, 3, ...)
- `title`: string - the scene's title
- `narration`: string - the voiceover script for the scene

Critical rules
1. Output ONLY the JSON. No explanations, no markdown fences, no "Here's the script:".

2. id must be 1, 2, 3, ... in order

3. Each scene's narration + visual must make sense TOGETHER.

4. LaTeX uses double backslashes in JSON: $\\\\sigma$ becomes $\\sigma$ when parsed.

5. NEVER use placeholders. Write the actual math equations.
"""

SCRIPTWRITER_USER_TEMPLATE = """
Topic: {query}

Write a complete script with 3-5 scenes. Output ONLY the JSON object.
The JSON must be like this: {{
    "title": "string",
    "scenes": [
        {{
            "id": 1,
            "title": "string",
            "narration": "string"
        }}
    ]
}}"""


def _count_words(text: str) -> int:
    return len(text.split())

def _validate_script_quality(script: ScriptOutput) -> tuple[bool, List[str]]:
    # Return (is_valid, list_of_warnings).

    warnings = []

    # Check 1: min 3 scenes, max 6 scenes
    if len(script.scenes) < 3:
        warnings.append(f"Too few scenes: {len(script.scenes)} (minimum 3)")

    if len(script.scenes) > 6:
        warnings.append(f"Too many scenes: {len(script.scenes)} (maximum 6)")

    # Check 2: scene IDs must be consecutive starting from 1
    expected_ids = list(range(1, len(script.scenes) + 1))
    actual_ids = [s.id for s in script.scenes]

    if actual_ids != expected_ids:
        warnings.append(f"Scene IDs not consecutive: {actual_ids}")

    # Check 3: Each scene's title and narration must be non-empty and of reasonable length
    for scene in script.scenes:
        if not scene.narration or _count_words(scene.narration) < 10:
            warnings.append(f"Scene {scene.id}: narration too short ({_count_words(scene.narration)} words)")

        if _count_words(scene.narration) > 100:
            warnings.append(f"Scene {scene.id}: narration too long ({_count_words(scene.narration)} words)")

    return len(warnings) == 0, warnings



def _make_fallback_script(query: str) -> ScriptOutput:
    return ScriptOutput(
        title=f"Explanation of {query[:50]}",
        scenes=[
            {
                "id": 1,
                "title": "Introduction",
                "narration": f"Today we'll explore {query}. This is a fundamental concept that's worth understanding deeply.",
            },
            {
                "id": 2,
                "title": "Core Concept",
                "narration": f"The main idea behind {query} can be understood by looking at its key components and how they interact with each other.",
            },
            {
                "id": 3,
                "title": "Summary",
                "narration": f"To summarize, {query} is a concept that helps us understand a specific aspect of this field. With this foundation, you can explore further applications.",
            },
        ],
    )

def make_scriptwriter_node(llm: LLM, max_retries: int = 2):
    parser = PydanticOutputParser(pydantic_object=ScriptOutput)

    def scriptwriter_node(state: AgentState) -> AgentState:
        query = state["user_query"]

        script = None
        last_error = None

        # Retry loop
        for attempt in range(max_retries + 1):
            try:
                # On retry, add extra instructions to the prompt to help the LLM correct its output
                if attempt > 0 and last_error:
                    extra_instruction = f"\n\n## PREVIOUS ATTEMPT FAILED\nError: {last_error}\nPlease ensure your output is valid JSON matching the schema exactly."
                else:
                    extra_instruction = ""

                user_prompt = SCRIPTWRITER_USER_TEMPLATE.format(
                    query=query,
                ) + extra_instruction

                # Get the LLM response
                response = llm.generate(system_instruction = SCRIPTWRITER_SYSTEM_PROMPT, prompt = user_prompt)
                
                # Parse the response
                script = parser.parse(response)

                # Validate quality of the script
                is_valid, warnings = _validate_script_quality(script)

                if is_valid:
                    break
                else:
                    if attempt < max_retries:
                        last_error = f"Quality issues: {'; '.join(warnings)}"
                        continue
                    else:
                        break

            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                
                if attempt < max_retries:
                    continue
                else:
                    break

        # Fallback if all attempts failed
        if script is None:
            script = _make_fallback_script(query)

            state["errors"] = state.get("errors", []) + [
                f"Scriptwriter: all attempts failed, used fallback. Last error: {last_error}"
            ]

        # Update state
        state["script"] = script.model_dump()
        state["scenes"] = [scene.model_dump() for scene in script.scenes]
        state["current_scene_idx"] = 0
        state["scene_codes"] = {}  # reset scene codes

        return state

    return scriptwriter_node