# backend/agent/nodes/planner.py
"""
Planner node: primește user query, returnează un plan structurat.
"""

from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from backend.agent.state import AgentState, PlanOutput, PlanStep


# ═══════════════════════════════════════════════════════════════════
# PROMPT
# ═══════════════════════════════════════════════════════════════════

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a video production planner for an educational platform.

Given a topic, create a step-by-step plan to produce a short animated explainer video (1-3 minutes).

## Available tools
- `kb_search`: Search the user's local knowledge base (uploaded documents)
- `web_search`: Search the web for current information
- `generate_manim`: Generate Python code for a Manim animation scene
- `generate_tts`: Generate voiceover audio from text
- `assemble_video`: Concatenate video clips and audio into final video

## Standard plan structure (4-6 steps)

For MOST topics, follow this pattern:
1. **research** — Gather information (use kb_search if user has uploaded documents, web_search for current topics)
2. **outline** — Plan the structure (no tool needed, just internal planning)
3. **script** — Write the narration script (no tool needed, just internal writing)
4. **render_manim** — Generate Manim animation code for each scene (use generate_manim)
5. **narrate** — Generate voiceover for each scene (use generate_tts)
6. **assemble** — Combine into final video (use assemble_video)

## When to deviate
- For a topic the LLM knows well (e.g., "What is backpropagation?"), skip research
- For a topic requiring current info (e.g., "Latest AMD GPUs in 2026"), use web_search
- For a personal note (e.g., "Explain my lecture notes on X"), use kb_search

## Output format
Return a JSON object with a `steps` array. Each step has:
- `step`: short name (snake_case)
- `description`: what this step does (1 sentence)
- `tools`: array of tool names needed (empty array if none)

## Example output for "Explain self-attention"
```json
{{
  "steps": [
    {{"step": "research", "description": "Search knowledge base for self-attention concepts and formulas", "tools": ["kb_search"]}},
    {{"step": "outline", "description": "Plan 4 scenes: intro, definition, example, summary", "tools": []}},
    {{"step": "script", "description": "Write narration for each scene with clear explanations", "tools": []}},
    {{"step": "render_manim", "description": "Generate Manim code for each scene and render to video", "tools": ["generate_manim"]}},
    {{"step": "narrate", "description": "Synthesize voiceover for each scene's narration", "tools": ["generate_tts"]}},
    {{"step": "assemble", "description": "Concatenate scene videos with audio into final video", "tools": ["assemble_video"]}}
  ]
}}
```"""),
    ("user", "Topic: {query}\n\nCreate a plan."),
])


# ═══════════════════════════════════════════════════════════════════
# Fallback plan (dacă LLM-ul eșuează)
# ═══════════════════════════════════════════════════════════════════

DEFAULT_PLAN = [
    {"step": "research", "description": "Gather information about the topic", "tools": ["kb_search", "web_search"]},
    {"step": "outline", "description": "Plan the video structure with 3-5 scenes", "tools": []},
    {"step": "script", "description": "Write narration script for each scene", "tools": []},
    {"step": "render_manim", "description": "Generate and render Manim animation for each scene", "tools": ["generate_manim"]},
    {"step": "narrate", "description": "Generate voiceover for each scene", "tools": ["generate_tts"]},
    {"step": "assemble", "description": "Combine scenes and audio into final video", "tools": ["assemble_video"]},
]


# ═══════════════════════════════════════════════════════════════════
# Nodul propriu-zis
# ═══════════════════════════════════════════════════════════════════

def make_planner_node(llm):
    """
    Factory: creează nodul planner cu LLM-ul capturat.
    
    Usage:
        planner_node = make_planner_node(llm)
        graph.add_node("planner", planner_node)
    """
    parser = PydanticOutputParser(pydantic_object=PlanOutput)
    
    def planner_node(state: AgentState) -> AgentState:
        query = state["user_query"]
        logger.info(f"🧠 Planning for: '{query[:80]}{'...' if len(query) > 80 else ''}'")
        
        try:
            # 1. Construiește prompt-ul
            messages = PLANNER_PROMPT.format_messages(query=query)
            
            # 2. Apelează LLM
            print("AICI: ------------------------------", messages)
            response = llm.generate(messages)

            response_text = response.content if hasattr(response, 'content') else str(response)
            
            logger.debug(f"Planner LLM response:\n{response_text[:500]}")
            
            # 3. Parse output ca Pydantic
            plan_output = parser.parse(response_text)
            
            # 4. Serialize pentru state
            state["plan"] = [step.model_dump() for step in plan_output.steps]
            state["current_plan_step"] = 0
            
            logger.success(f"✅ Plan created: {len(state['plan'])} steps")
            for i, step in enumerate(state["plan"]):
                tools_str = f" (tools: {', '.join(step['tools'])})" if step['tools'] else ""
                logger.info(f"   {i+1}. {step['step']}{tools_str}")
                logger.debug(f"      → {step['description']}")
        
        except Exception as e:
            logger.error(f"❌ Planner failed: {e}")
            logger.debug(f"Raw response: {response_text if 'response_text' in locals() else 'N/A'}")
            
            # Fallback: plan default
            state["plan"] = DEFAULT_PLAN
            state["current_plan_step"] = 0
            state["errors"] = state.get("errors", []) + [f"Planner error: {e}"]
            logger.warning("⚠️ Using default plan as fallback")
        
        return state
    
    return planner_node