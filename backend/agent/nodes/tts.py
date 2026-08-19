from pathlib import Path
from typing import Any, Dict

from backend.agent.state import AgentState
from tools.tts.synthesize import Synthesizer, SynthesisError, _wav_duration


def _synthesize(
    synthesizer: Synthesizer,
    text: str,
    output_path: str,
    voice: str = "narrator",
    language: str = "en",
) -> Dict[str, Any]:
    # Synthesize the text to speech and save to output_path
    # Returns {"success": bool, "audio_path": str, "duration_sec": float, "error": str|None}

    try:
        print("Calling TTS engine with text:", text, "\n\n", "Voice: ", voice, flush=True)

        audio_path = synthesizer.synthesize(text, output_path, voice=voice, language=language)

        print("TTS engine returned:", audio_path, "\n\n", flush=True)

        return {
            "success": True,
            "audio_path": audio_path,
            "duration_sec": _wav_duration(audio_path),
            "error": None,
        }
    except SynthesisError as exc:
        return {
            "success": False,
            "audio_path": None,
            "duration_sec": 0,
            "error": str(exc),
        }
    except Exception as e:
        return {
            "success": False,
            "audio_path": None,
            "duration_sec": 0,
            "error": f"{type(e).__name__}: {str(e)}",
        }

def _synthesize_scene(output_path: Path, synthesizer: Synthesizer, voice: str, language: str, scene: dict) -> tuple[int, dict]:
    scene_id = scene.get("id", 0)
    scene_title = scene.get("title", f"Scene {scene_id}")
    narration = scene.get("narration", "").strip()
    
    if not narration:
        return scene_id, {
            "success": False,
            "audio_path": None,
            "duration_sec": 0,
            "error": "Empty narration",
        }
    
    audio_file = str(output_path / f"narration_{scene_id}.wav")

    print(f"Synthesizing TTS for scene {scene_id} ('{scene_title}') to '{audio_file}'...\n\n", flush=True)
    
    result = _synthesize(synthesizer, narration, audio_file, voice, language)

    print(scene_id, scene_title, result, "\n\n", flush=True)
    
    return scene_id, result


def make_tts_node(
    output_dir: str = "./output/audio",
    voice: str = "narrator",
    language: str = "en"
):  
    def tts_node(state: AgentState) -> AgentState:
        scenes = state.get("scenes", [])
        
        if not scenes:
            return state

        print(f"Starting TTS synthesis for {len(scenes)} scenes with voice '{voice}' and language '{language}'...\n\n", flush=True)
                
        # Setup output dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Initializing TTS engine once for all scenes...\n\n", flush=True)
        synthesizer = Synthesizer()
        print("TTS engine initialized.\n\n", flush=True)        
        
        results = []
        for scene in scenes:
            try:
                res = _synthesize_scene(output_path, synthesizer, voice, language, scene)
                results.append(res)
            except Exception as e:
                results.append(e)
        
        # Process results and update state
        audio_tracks = state.get("audio_tracks", {})
        errors = state.get("errors", [])
        
        for result in results:
            if isinstance(result, Exception):
                errors.append(f"TTS task exception: {result}")
                continue

            # Assign audio track if successful
            scene_id, tts_result = result
            if tts_result["success"] and tts_result["audio_path"]:
                audio_tracks[scene_id] = {"path": tts_result["audio_path"], "duration_sec": tts_result["duration_sec"]}
            else:
                errors.append(
                    f"TTS scene {scene_id} failed: {tts_result.get('error', 'unknown')}"
                )
        
        # Update state
        state["audio_tracks"] = audio_tracks
        state["errors"] = errors
        
        return state
    
    return tts_node
