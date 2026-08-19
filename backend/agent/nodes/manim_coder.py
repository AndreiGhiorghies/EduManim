import json
import subprocess
from pathlib import Path
from typing import Optional

from LLM.llm import LLM
from backend.agent.state import AgentState

from moviepy import VideoFileClip

MANIM_CODER_SYSTEM_PROMPT2 = """
You are an expert in Manim Community v0.20.1. Generate strictly clean, working Python code for ONE scene based on the narration.

# REQUIREMENTS

## Imports & Class
- Use ONLY: `from manim import *`
- Define exactly ONE class inheriting from `Scene`, named `Scene{scene_id}`.
- Implement `def construct(self):` with all logic inside.

## Allowed Constructs
- Mobjects: Text, MathTex, Tex, MarkupText, Paragraph, Code, Circle, Square, Rectangle, Polygon, RegularPolygon, Arrow, DoubleArrow, Line, Dot, VGroup
- Layout: UP, DOWN, LEFT, RIGHT, ORIGIN, UL, UR, DL, DR
- Colors: WHITE, BLACK, RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, PINK, GREY, TEAL
- Animations: Create, Write, FadeIn, FadeOut, Transform, ReplacementTransform, GrowFromCenter, Indicate
- Composition: AnimationGroup, Succession, Wait
- Math: MathTex, Tex (MUST ALWAYS use raw strings: e.g., MathTex(r"\frac{1}{2}") )
- Positioning: .to_edge(), .move_to(), .next_to(), .align_to()
- Grouping: VGroup (Use VGroup().arrange(DOWN) to prevent overlapping of multiple elements)
- Modifiers: .scale(), .set_color(), .set_opacity()

## STRICT Constraints
- NO SVGMobject, NO ThreeDScene, NO updaters.
- NO absolute manual coordinates (e.g., avoid LEFT * 3). Rely on .next_to() and .arrange() to prevent overlaps.
- NO external imports (os, sys, requests, etc.).
- ALWAYS prefix MathTex and Tex strings with 'r' (raw string).
- Make sure the animations are fluid and match the narration timing. Avoid abrupt transitions or long pauses.
- Be careful to not overlap text and unwanted elements, and that the positioning is visually appealing, and if you don't use an element anymore remove it.

## Timing
- Use run_time= for animations.
- Use self.wait(1) to separate distinct visual ideas.
- Total scene duration: time length needs to be exactly {scene_duration}.
- The animations should move all along with the narration, so make sure the timing of animations matches the narration positions. The narrator reads around 2 words per second, make sure that the animations are not too fast or too slow compared to the narration and they keep track of the narration.

# OUTPUT FORMAT
Return strictly valid PYTHON CODE only.

Put the Python code inside a fenced block, for example:
```python\nfrom manim import *\n...\n```

The code must contain only the Manim code needed for the scene.
The code must start with "from manim import *" after the fence is removed.

OUTPUT ONLY THE VALID PYTHON CODE, NOTHING ELSE, NOT ANY WORDS FROM THOUGHT PROCESS SHOULD APPEAR OUTSIDE THE CODE.
"""


MANIM_CODER_SYSTEM_PROMPT = """
You are an expert in Manim Community v0.20.1. Generate strictly clean, working Python code for ONE scene based on the narration.

# REQUIREMENTS

## Imports & Class
- Use ONLY: `from manim import *`
- Define exactly ONE class inheriting from `Scene`, named `Scene{scene_id}`.
- Implement `def construct(self):` with all logic inside.

## Visual Design (SHOW, DON'T TELL)
- MAXIMIZE visual elements (shapes, diagrams, abstract representations using VGroup).
- MINIMIZE text on screen. DO NOT just write the narration on the screen.
- Use `Text` or `MarkupText` ONLY for short titles, key terms, or labels.
- Build complex, appealing visuals by combining basic shapes (Circle, Rectangle, Line, Arrow) and animating them fluidly.

## Allowed Constructs
- Mobjects: Text, MathTex, Tex, MarkupText, Paragraph, Code, Circle, Square, Rectangle, Polygon, RegularPolygon, Arrow, DoubleArrow, Line, Dot, VGroup
- Layout: UP, DOWN, LEFT, RIGHT, ORIGIN, UL, UR, DL, DR
- Colors: WHITE, BLACK, RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, PINK, GREY, TEAL
- Animations: Create, Write, FadeIn, FadeOut, Transform, ReplacementTransform, GrowFromCenter, Indicate
- Composition: AnimationGroup, Succession, Wait
- Positioning: .to_edge(), .move_to(), .next_to(), .align_to()
- Grouping: VGroup (Use VGroup().arrange(DOWN) to prevent overlapping of multiple elements)
- Modifiers: .scale(), .set_color(), .set_opacity()

## STRICT Constraints
- NO SVGMobject, NO ThreeDScene, NO updaters.
- NO absolute manual coordinates (e.g., avoid LEFT * 3).
- NEVER invent properties for coordinates (DO NOT use .center_left, etc.). ALWAYS use .get_left(), .get_right(), .get_center().
- NO external imports (os, sys, requests, etc.).
- ALWAYS prefix MathTex and Tex strings with 'r' (raw string).
- Remove elements from the screen when they are no longer relevant to the narration.
- Make sure the animations are fluid and match the narration timing. Avoid abrupt transitions or long pauses.
- Make sure the positioning is visually appealing and that text do not overlap.
- Be cafeful that the end of the animation is smooth and that the narration has a smooth end.

## Timing & Synchronization
- Total scene duration needs to be EXACTLY {scene_duration} seconds.
- The narrator reads at ~2 words per second. 
- You MUST break down the narration sentence by sentence.
- Add Python comments (e.g., `# Narration: "Imagine speaking into a device..." (~3 seconds)`) before each animation block to explicitly show your timing logic.
- Use `run_time=...` on `self.play()` and `self.wait(...)` to perfectly pad the time so the visual actions map exactly to the words being spoken.

# OUTPUT FORMAT
Return strictly valid PYTHON CODE ONLY. DO NOT output JSON. DO NOT output any reasoning outside the code block.

Put the Python code inside a fenced markdown block:
```python
from manim import *
...
OUTPUT ONLY THE FENCED PYTHON CODE.
"""


MANIM_CODER_USER_TEMPLATE2 = """Scene {scene_id}: {scene_title}

Visual hint: {visual_hint}

Narration(for context): {narration}

{feedback_section}

Generate the scene response as PYTHON CODE ONLY.
The code must be inside a fenced Python block (```python ... ```).
DO NOT output any thoughts, JSON, or text outside the code block.

MAKE SURE THE ANIMATIONS ARE FLUID AND HAVE A GOOD POSITIONING AND THAT THE ANIMATIONS ARE VISUALLY APPEALING EVEN IF YOU NEED TO GENERATE MUCH MORE CODE FOR IT.
THE PYTHON CODE SHOULD REFLECT ALSO THE NARRATION TIMING, AND THE ANIMATIONS SHOULD MATCH THE NARRATION TIMING AND THE THINGS THAT ARE SAID IN THE NARRATION.
OUTPUT ONLY THE VALID PYTHON CODE.
"""


MANIM_CODER_USER_TEMPLATE = """Scene {scene_id}: {scene_title}

Narration (Total Duration: {scene_duration}s): 
"{narration}"

{feedback_section}

Generate the scene response as PYTHON CODE ONLY.
The code must be inside a fenced Python block (```python ... ```).

CRITICAL INSTRUCTIONS:
1. FOCUS ON VISUALS: Draw diagrams, icons, or abstract concepts using geometric shapes. Do NOT just print the narration as Text.
2. SYNCHRONIZE EXACTLY: Place fragments of the narration as comments before the respective `self.play()` calls. 
3. Calculate your `run_time` and `self.wait()` so the sum of all durations is exactly {scene_duration} seconds.

OUTPUT ONLY THE VALID PYTHON CODE.
"""

def _extract_code(response: str) -> str:
    response = response.strip()

    if response.startswith("```") and response.endswith("```"):
        code_content = response[3:-3].strip()
        if code_content.startswith("python"):
            code_content = code_content[6:].strip()
        return code_content

    return response.strip()


# Save the generated code and return the path
def _save_code_to_file(code: str, output_dir: str, scene_id: int) -> str:
    output_path = Path(output_dir + "/code")
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / f"scene_{scene_id}.py"
    file_path.write_text(code, encoding="utf-8")

    return str(file_path)

# Validate the generated code
def _validate_code(scene_file: str) -> tuple[bool, str]:
    # Returns (success, error_message)
    try:
        result = subprocess.run(
            ["python3", "-m", "tools.manim.cli", "validate", scene_file],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr.strip() or result.stdout.strip()
    
    except subprocess.TimeoutExpired:
        return False, "Validation timeout"
    except Exception as e:
        return False, f"Validator error: {e}"


# Render the generated code
# Returns (success, error_message, json_output)
def _render_scene(
    scene_file: str, 
    scene_class_name: str, 
    output_video: str
) -> tuple[bool, str, Optional[dict[str, object]]]:
    try:
        result = subprocess.run(
            [
                "python3", "-m", "tools.manim.cli", "render",
                scene_file, scene_class_name,
                "--output", output_video,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if result.returncode == 0:
            try:
                output = json.loads(result.stdout)
                if output.get("success"):
                    return True, "", output
                else:
                    return False, output.get("error", "Unknown render error"), None
            except json.JSONDecodeError:
                if Path(output_video).exists():
                    return True, "", {"video_path": output_video, "duration_sec": 0}
                return False, f"Invalid output from renderer: {result.stdout[:200]}", None
        else:
            return False, result.stderr.strip() or result.stdout.strip(), None
    
    except subprocess.TimeoutExpired:
        return False, "Render timeout (120s)", None
    except Exception as e:
        return False, f"Renderer error: {e}", None


# Fallback code generation for a simple static scene
def _generate_fallback_code(scene_id: int, narration: str, duration: int) -> str:
    text = narration[:80].replace('"', "'").replace("\n", " ")
    
    return f'''
from manim import *

class Scene{scene_id}(Scene):
    def construct(self):
        # Fallback static scene
        title = Text("{scene_id}. Scene", font_size=36, color=BLUE)
        title.to_edge(UP)
        
        body = Text("{text}", font_size=20, color=WHITE)
        body.move_to(ORIGIN)
        body.width = min(body.width, 12)
        
        self.play(Write(title))
        self.play(FadeIn(body))
        self.wait({duration})
        self.play(FadeOut(title), FadeOut(body))
'''

# Render a fallback scene if all attempts fail
def _generate_fallback_render(
    scene_id: int,
    narration: str,
    output_dir: str,
    duration: int
) -> tuple[bool, str, Optional[dict[str, object]]]:
    code = _generate_fallback_code(scene_id, narration, duration)
    scene_file = _save_code_to_file(code, output_dir, scene_id)
    scene_class = f"Scene{scene_id}"
    output_video = str(Path(output_dir + "/mp4") / f"scene_{scene_id}.mp4")
    
    return _render_scene(scene_file, scene_class, output_video)

def _generate_scene(llm, output_dir: str, state: AgentState, max_retries: int) -> AgentState:
    scenes = state.get("scenes", [])
    current_idx = state.get("current_scene_idx", 0)
    
    if current_idx >= len(scenes):
        return state
    
    scene = scenes[current_idx]
    scene_id = scene.get("id", current_idx + 1)
    scene_title = scene.get("title", f"Scene {scene_id}")
    narration = scene.get("narration", "")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Retry loop: generate -> validate -> render
    code = None
    video_path: Optional[str] = None
    last_error = None
    used_fallback = False
    duration = state["audio_tracks"].get(scene_id, {}).get("duration_sec", 10)
    
    for attempt in range(max_retries + 1):
        try:                
            # Generate the code
            if attempt == 0:
                feedback = ""
            else:
                feedback = f"\n\nPREVIOUS ATTEMPT FAILED:\n{last_error}\n\nPlease fix the code and try again."
            
            user_prompt = MANIM_CODER_USER_TEMPLATE.format(
                scene_id=scene_id,
                scene_title=scene_title,
                scene_duration=duration,
                narration=narration,
                feedback_section=feedback,
            )
            
            system_prompt = MANIM_CODER_SYSTEM_PROMPT.replace("{scene_id}", str(scene_id)).replace("{scene_duration}", str(duration))
            
            response = llm.generate(
                system_instruction=system_prompt,
                prompt=user_prompt,
                temperature=0.3,
            )
            
            code = _extract_code(response)
            
            # Save the code to file
            scene_file = _save_code_to_file(code, output_dir, scene_id)
            
            # Validate the code
            is_valid, val_error = _validate_code(scene_file)
            if not is_valid:
                last_error = f"Validation failed: {val_error}"
                if attempt < max_retries:
                    continue
                else:
                    break
                            
            # Render scene
            output_video = str(Path(output_dir + "/mp4") / f"scene_{scene_id}.mp4")
            success, render_error, render_output = _render_scene(
                scene_file, f"Scene{scene_id}", output_video
            )

            if not success:
                last_error = f"Render failed: {render_error}"
                if attempt < max_retries:
                    continue
                else:
                    break

            # Check duration to match with the duration of the audio
            if render_output is None:
                raise RuntimeError("Renderer did not return metadata")

            rendered_video_path = str(render_output.get("video_path", output_video))
            duration_real = 0
            with VideoFileClip(rendered_video_path) as clip:
                duration_real = clip.duration

            print(f"Scene {scene_id} rendered. Expected duration: {duration:.2f}s, actual: {duration_real:.2f}s.\n\n")

            if abs(duration - duration_real) > 1:
                last_error = f"Duration mismatch: expected {duration}, got {duration_real}. Make sure the animations match the narration timing ({duration} seconds)."
                if attempt < max_retries:
                    continue
                else:
                    break

            video_path = rendered_video_path
            break
        
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            if attempt < max_retries:
                continue
            else:
                break
    
    # Fallback if all attempts fail
    if video_path is None:
        used_fallback = True
        
        success, fb_error, fb_output = _generate_fallback_render(
            scene_id, narration, output_dir, duration
        )
        
        if success and fb_output:
            video_path = str(fb_output.get("video_path", ""))
            code = _generate_fallback_code(scene_id, narration, duration)
        else:
            state["errors"] = state.get("errors", []) + [
                f"Manim Coder scene {scene_id}: all attempts + fallback failed. Last: {last_error}, Fallback: {fb_error}"
            ]
            video_path = None
    
    # Update the state for this scene
    scene_codes = state.get("scene_codes", {})
    scene_videos = state.get("scene_videos", {})
    
    if code:
        scene_codes[scene_id] = code
    if video_path:
        scene_videos[scene_id] = video_path
    
    state["scene_codes"] = scene_codes
    state["scene_videos"] = scene_videos
    state["current_scene_idx"] = current_idx + 1
    state["needs_retry"] = False
    state["last_error"] = None
    state["retries_for_current_scene"] = 0
    state["used_fallback"] = state.get("used_fallback", False) or used_fallback
    
    return state

def make_manim_coder_node(llm: LLM, output_dir: str, max_retries: int = 2):
    
    def manim_coder_node(state: AgentState) -> AgentState:
        total = len(state.get("scenes", []))

        for i in range(total):
            print(f"Processing scene {i + 1}/{total}...")
    
            state = _generate_scene(llm, output_dir, state, max_retries)
            
            scene_id = state["scenes"][i].get("id", i + 1)
            video = state["scene_videos"].get(scene_id)

            print(f"Scene {i + 1}/{total} done. Video: {video}\n\n")
            
        return state
    
    return manim_coder_node
