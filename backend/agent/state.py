from typing import TypedDict, Optional, List, Dict, Any
from pydantic import BaseModel, Field

class Scene(BaseModel):
    id: int = Field(description="Scene ID, starting from 1")
    title: str = Field(description="Short title for the scene")
    narration: str = Field(
        description="Text spoken for voiceover"
    )

class ScriptOutput(BaseModel):
    title: str = Field(description="Title of the video")
    scenes: List[Scene] = Field(
        description="3-5 scenes that form the video"
    )
    thought_process: Optional[str] = Field(
        default=None,
        description="Internal reasoning of the LLM: Use this to calculate word counts, durations, and scene limits before finalizing scenes",
    )

class AgentState(TypedDict):
    # The state of the agent. All nodes receive and return this structure.
    
    # Input
    user_query: str
    
    # Script
    script: Optional[Dict[str, Any]]
    scenes: List[Dict[str, Any]]
    current_scene_idx: int
    
    # Code Generation
    scene_codes: Dict[int, str]          # scene_id -> Python code
    needs_retry: bool
    last_error: Optional[str]
    retries_for_current_scene: int
    used_fallback: bool
    
    # Rendering
    scene_videos: Dict[int, str]             # scene_id -> video file path
    audio_tracks: Dict[int, Dict[str, Any]]  # scene_id -> { "path": audio file path, "duration_sec": duration }
    
    # Final output
    final_video_path: Optional[str]
    
    # Metadata
    errors: List[str]
    started_at: Optional[str]


def create_initial_state(
    user_query: str,
    kb_id: Optional[str] = None,
) -> AgentState:
    from datetime import datetime, timezone
    
    return {
        "user_query": user_query,
        "kb_id": kb_id,
        
        "research_notes": [],
        
        "script": None,
        "scenes": [],
        "current_scene_idx": 0,
        
        "scene_codes": {},
        "needs_retry": False,
        "last_error": None,
        "retries_for_current_scene": 0,
        "used_fallback": False,
        
        "scene_videos": {},
        "audio_tracks": {},
        "final_video_path": None,
        
        "errors": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }