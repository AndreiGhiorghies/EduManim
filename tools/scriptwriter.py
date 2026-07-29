from typing import List

from langchain_core.output_parsers import PydanticOutputParser


from backend.agent.state import AgentState, ScriptOutput


# PROMPT

SCRIPTWRITER_SYSTEM_PROMPT = """You are an expert educational scriptwriter specializing in short animated explainer videos (60-180 seconds total).


## Your task

Given a topic and optional research notes, write a structured video script with 3-5 scenes.


## Scene structure pattern

A good script follows this arc:

1. **Hook** (10-20s): Pose the question or problem. Why should I care?

2. **Core concept** (30-60s): The main idea, with formula/diagram

3. **Example** (30-60s): Concrete walkthrough showing it in action

4. **Recap** (10-20s): Summarize key points


## For each scene, you must provide


**title**: 3-5 words, like "What is Attention?" or "The Backprop Formula"


**narration** (1-3 sentences, 20-80 words):

- Spoken by voiceover, so write CONVERSATIONALLY

- Explain clearly for a student seeing it for the first time

- Define jargon when first used

- Use transitions to connect scenes ("Now that we know X, let's see Y")

- Read it aloud mentally — does it sound natural?


**visual_hint** (concrete, specific):

- Use LaTeX for math: `$\\sigma(x)$`, `$x^2$`, `$\\frac{{a}}{{b}}$`

- Specify colors, positions, animations

- BAD: "Show a diagram"

- GOOD: "Title 'Self-Attention' fades in. Below: matrix Q, K, V appear as colored boxes. Arrow from Q to 'softmax' box. Result fades in as heatmap."

- Be specific enough that a Manim developer can implement it


**key_concepts** (2-4 items): Main ideas covered, for verification


## Output format

JSON object with:

- `title`: string

- `estimated_duration_sec`: integer (sum of scenes at ~150 words/min)

- `scenes`: array of 3-5 Scene objects (id starts at 1)


## Example 1: Technical topic

Topic: "Self-Attention"

```json

{{

  "title": "Self-Attention Explained",

  "estimated_duration_sec": 120,

  "scenes": [

    {{

      "id": 1,

      "title": "Why Attention Matters",

      "narration": "When you read a sentence, you don't process each word alone. Your brain automatically focuses on relevant words to understand context. Self-attention is how AI models do the same thing.",

      "visual_hint": "Title 'Self-Attention' fades in. A sentence 'The cat sat on the mat' appears word by word. Curved arrows connect 'cat' to 'sat' and 'mat' showing relationships, animating in sequence.",

      "key_concepts": ["attention", "context understanding", "word relationships"]

    }},

    {{

      "id": 2,

      "title": "The Attention Formula",

      "narration": "Self-attention computes three vectors for each word: Query, Key, and Value. The attention score is the dot product of Query and Key, normalized by a softmax function.",

      "visual_hint": "Center the formula $\\text{{Attention}}(Q, K, V) = \\text{{softmax}}(\\frac{{QK^T}}{{\\sqrt{{d_k}}}})V$ in white. Three colored boxes labeled Q (red), K (green), V (blue) appear on the left. Arrows flow through operations, ending in a softmax result box on the right.",

      "key_concepts": ["query", "key", "value", "softmax", "dot product"]

    }},

    {{

      "id": 3,

      "title": "Walking Through an Example",

      "narration": "Let's trace through our sentence. The word 'cat' acts as a Query and computes attention scores with all other words. It gets high scores for 'sat' and 'mat' because they're contextually related.",

      "visual_hint": "Show the sentence with 'cat' highlighted in red. A 5x5 attention heatmap matrix appears below, with the 'cat' row showing bright cells at positions 2 ('sat') and 4 ('mat'). Animated arrows connect 'cat' to these words.",

      "key_concepts": ["attention weights", "contextual relationships", "visualization"]

    }},

    {{

      "id": 4,

      "title": "Recap",

      "narration": "Self-attention lets each word look at all other words and decide which matter most. It's the core mechanism behind transformers and modern language models.",

      "visual_hint": "Summary text 'Self-Attention: Each word looks at all others' appears center-screen. Fade to title card 'Self-Attention Explained' with subtitle 'Thanks for watching!'",

      "key_concepts": ["summary", "transformers"]

    }}

  ]

}}

```


## Example 2: Conceptual topic

Topic: "Why the Sky is Blue"

```json

{{

  "title": "Why the Sky is Blue",

  "estimated_duration_sec": 90,

  "scenes": [

    {{

      "id": 1,

      "title": "A Colorful Mystery",

      "narration": "Look up on a clear day. The sky is blue, sunsets are red, and clouds are white. But why? It all comes down to how light interacts with air.",

      "visual_hint": "Title 'Why is the Sky Blue?' fades in. Below: a split scene — bright blue sky on left, red sunset on right, white clouds in middle. Sun rays enter from top.",

      "key_concepts": ["light", "atmosphere", "color perception"]

    }},

    {{

      "id": 2,

      "title": "Sunlight: A Rainbow Mix",

      "narration": "Sunlight looks white, but it's actually a mix of all colors. Each color is a different wavelength of light, traveling in waves.",

      "visual_hint": "White light beam enters from left. A prism in the middle splits it into a rainbow spectrum (red, orange, yellow, green, blue, violet) fanning out to the right. Label each color with its wavelength in nanometers.",

      "key_concepts": ["wavelength", "visible spectrum", "white light"]

    }},

    {{

      "id": 3,

      "title": "Rayleigh Scattering",

      "narration": "When light hits air molecules, it scatters. Blue light scatters ten times more than red because it has a shorter wavelength. So when you look up, you see scattered blue light coming from every direction.",

      "visual_hint": "Show air molecules as small dots. A blue light ray hits a dot and scatters in many directions (arrows). A red light ray passes through mostly unaffected. Equation $\\text{{scattering}} \\propto \\frac{{1}}{{\\lambda^4}}$ appears in corner.",

      "key_concepts": ["Rayleigh scattering", "wavelength dependence", "molecular interaction"]

    }},

    {{

      "id": 4,

      "title": "Why Sunsets are Red",

      "narration": "At sunset, light travels through more air to reach you. Most blue light scatters away before it gets to you, leaving only the red and orange wavelengths. That's why sunsets are red.",

      "visual_hint": "Sun near horizon. Light ray from sun to observer passes through thick layer of atmosphere (shown as many dots). Blue scatters out along the way, red makes it through to the observer. Sky around sun shows red/orange gradient.",

      "key_concepts": ["atmospheric path length", "color filtering", "sunset physics"]

    }}

  ]

}}

```


## Critical rules

1. Output ONLY the JSON. No explanations, no markdown fences, no "Here's the script:".

2. `id` must be 1, 2, 3, ... in order

3. `estimated_duration_sec` must be plausible (60-180 for typical 3-5 scene scripts)

4. Each scene's narration + visual must make sense TOGETHER

5. LaTeX uses double backslashes in JSON: `$\\\\sigma$` becomes `$\\sigma$` when parsed

6. Don't use placeholder text like "[insert formula here]" — be specific


## If you don't know

If the topic is unclear or you lack information, make reasonable assumptions for a general audience. Better to have a complete, somewhat generic script than an incomplete specific one."""



SCRIPTWRITER_USER_TEMPLATE = """Topic: {query}


{research_section}


Write a complete script with 3-5 scenes. Output ONLY the JSON object."""


# Helpers

def _format_research_notes(notes: List[str]) -> str:
    if not notes:
        return "## Research notes\nNone provided. Use your general knowledge."

    return "## Research notes\n" + "\n".join(f"- {n}" for n in notes)



def _count_words(text: str) -> int:
    return len(text.split())



def _estimate_narration_duration(text: str) -> int:
    words = _count_words(text)

    return max(5, int(words / 2.5))



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

    

    # Check 3: Each scene's title, narration, visual_hint must be non-empty and of reasonable length

    for scene in script.scenes:
        if not scene.title or len(scene.title) < 3:
            warnings.append(f"Scene {scene.id}: title too short")

        if not scene.narration or _count_words(scene.narration) < 10:
            warnings.append(f"Scene {scene.id}: narration too short ({_count_words(scene.narration)} words)")

        if _count_words(scene.narration) > 120:
            warnings.append(f"Scene {scene.id}: narration too long ({_count_words(scene.narration)} words)")

        if not scene.visual_hint or len(scene.visual_hint) < 20:
            warnings.append(f"Scene {scene.id}: visual_hint too vague")

    

    # Check 4: estimated_duration should have a reasonable length

    actual_total = sum(_estimate_narration_duration(s.narration) for s in script.scenes)

    if abs(script.estimated_duration_sec - actual_total) > 60:
        warnings.append(
            f"Duration estimate off: claimed {script.estimated_duration_sec}s, "

            f"actual ~{actual_total}s"
        )

    return len(warnings) == 0, warnings



def _make_fallback_script(query: str) -> ScriptOutput:
    return ScriptOutput(

        title=f"Explanation of {query[:50]}",

        estimated_duration_sec=60,

        scenes=[

            {

                "id": 1,

                "title": "Introduction",

                "narration": f"Today we'll explore {query}. This is a fundamental concept that's worth understanding deeply.",

                "visual_hint": f"Title text fades in: '{query[:40]}'. A simple, clean background. Then transition to the next scene.",

                "key_concepts": [query.lower()],

            },

            {

                "id": 2,

                "title": "Core Concept",

                "narration": f"The main idea behind {query} can be understood by looking at its key components and how they interact with each other.",

                "visual_hint": "Display the core concept in the center. Use simple shapes and labels. Add arrows to show relationships between components.",

                "key_concepts": ["main idea", "components"],

            },

            {

                "id": 3,

                "title": "Summary",

                "narration": f"To summarize, {query} is a concept that helps us understand a specific aspect of this field. With this foundation, you can explore further applications.",

                "visual_hint": "Summary text appears on screen with the key points. Fade to title card.",

                "key_concepts": ["summary"],

            },

        ],

    )


# Main scriptwriter node

def make_scriptwriter_node(llm, max_retries: int = 2):
    parser = PydanticOutputParser(pydantic_object=ScriptOutput)

    def scriptwriter_node(state: AgentState) -> AgentState:
        query = state["user_query"]
        research_notes = state.get("research_notes", [])

        script = None
        last_error = None

        # Retry loop
        for attempt in range(max_retries + 1):
            try:
                # Build the prompt with research notes
                research_section = _format_research_notes(research_notes)

                # On retry, add extra instructions to the prompt to help the LLM correct its output
                if attempt > 0 and last_error:
                    extra_instruction = f"\n\n## PREVIOUS ATTEMPT FAILED\nError: {last_error}\nPlease ensure your output is valid JSON matching the schema exactly."
                else:
                    extra_instruction = ""

                user_prompt = SCRIPTWRITER_USER_TEMPLATE.format(

                    query=query,

                    research_section=research_section,

                ) + extra_instruction

                # Get the LLM response
                response = llm.generate(system_instruction = SCRIPTWRITER_SYSTEM_PROMPT, prompt = user_prompt)
                
                # Parse ca Pydantic
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

        # POST-RETRY: fallback if all attempts failed
        if script is None:
            script = _make_fallback_script(query)

            state["errors"] = state.get("errors", []) + [
                f"Scriptwriter: all attempts failed, used fallback. Last error: {last_error}"
            ]

        # UPDATE STATE
        state["script"] = script.model_dump()

        state["scenes"] = [scene.model_dump() for scene in script.scenes]

        state["current_scene_idx"] = 0

        state["scene_codes"] = {}  # reset scene codes

        return state

    

    return scriptwriter_node