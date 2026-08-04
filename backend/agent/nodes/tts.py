import json
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from backend.agent.state import AgentState


# HELPERS

def _synthesize_sync(
    text: str,
    output_path: str,
    voice: str = "narrator",
    language: str = "en",
    timeout: int = 1800,
) -> Dict[str, Any]:
    """
    Synchronous call for Track B's TTS CLI.
    
    Returns:
        {"success": bool, "audio_path": str, "duration_sec": float, "error": str|None}
    """

    try:
        result = subprocess.run(
            [
                "python", "-m", "tools.tts.cli", "synthesize",
                text,
                "--output", output_path,
                "--voice", voice,
                "--language", language,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        if result.returncode != 0:
            # Try to parse JSON error output
            try:
                err_data = json.loads(result.stdout)
                return {
                    "success": False,
                    "audio_path": None,
                    "duration_sec": 0,
                    "error": err_data.get("error", result.stderr[:200]),
                }
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "audio_path": None,
                    "duration_sec": 0,
                    "error": result.stderr[:200] or result.stdout[:200],
                }
        
        # Succes — parse JSON output
        try:
            data = json.loads(result.stdout)
            return {
                "success": True,
                "audio_path": data.get("audio_path"),
                "duration_sec": data.get("duration_sec", 0),
                "error": None,
            }
        except json.JSONDecodeError:
            # Fallback: verify if output_path exists
            if Path(output_path).is_file():
                return {
                    "success": True,
                    "audio_path": output_path,
                    "duration_sec": 0,
                    "error": None,
                }
            return {
                "success": False,
                "audio_path": None,
                "duration_sec": 0,
                "error": "Invalid TTS output",
            }
    
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "audio_path": None,
            "duration_sec": 0,
            "error": f"TTS timeout ({timeout}s)",
        }
    except Exception as e:
        return {
            "success": False,
            "audio_path": None,
            "duration_sec": 0,
            "error": f"{type(e).__name__}: {str(e)}",
        }


def _list_voices_sync() -> list:
    """Obtains the list of available voices by calling the TTS CLI."""
    try:
        result = subprocess.run(
            ["python", "-m", "tools.tts.cli", "list-voices"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data if isinstance(data, list) else []
    except Exception as e:
        pass
    return []


# MAIN NODE

def make_tts_node(
    output_dir: str = "/tmp/edumanim/audio",
    voice: str = "narrator",
    language: str = "en",
    timeout_per_scene: int = 1800,
):  
    async def tts_node(state: AgentState) -> AgentState:
        scenes = state.get("scenes", [])
        
        if not scenes:
            return state
                
        # Setup output dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare async tasks for each scene
        async def synthesize_scene(scene: dict) -> tuple[int, dict]:
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
            
            # Call Track B in a thread pool (does not block event loop)
            result = await asyncio.to_thread(
                _synthesize_sync,
                narration,
                audio_file,
                voice,
                language,
                timeout_per_scene,
            )

            print(scene_id, scene_title, result, "\n\n")
            
            return scene_id, result
        
        # Run all scenes (sequentially because in parallel takes too much memory)
        results = []
        for scene in scenes:
            try:
                res = await synthesize_scene(scene)
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
