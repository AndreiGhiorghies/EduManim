import asyncio
import json
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

DB_PATH = Path("./data/edumanim.db")
MOCK_VIDEO_PATH = Path("./data/mock/final.mp4")
JOB_STATUSES = {"queued", "running", "completed", "failed", "cancelled"}

MOCK_STAGES = [
    ("planning", "Planning video structure..."),
    ("scripting", "Writing narration script..."),
    ("rendering_1", "Rendering scene 1/3..."),
    ("rendering_2", "Rendering scene 2/3..."),
    ("rendering_3", "Rendering scene 3/3..."),
    ("narration", "Synthesizing narration audio..."),
    ("assembling", "Assembling final video..."),
]
MOCK_STAGE_DELAY_SECONDS = 1.5


def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            user_query TEXT,
            status TEXT NOT NULL,
            plan_json TEXT,
            script_json TEXT,
            final_video_path TEXT,
            error TEXT,
            created_at TIMESTAMP,
            completed_at TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


class JobPubSub:
    def __init__(self):
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(job_id, []).append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue):
        subs = self._subscribers.get(job_id, [])
        if queue in subs:
            subs.remove(queue)

    async def publish(self, job_id: str, event: dict):
        for queue in self._subscribers.get(job_id, []):
            await queue.put(event)


pubsub = JobPubSub()
running_tasks: dict[str, asyncio.Task] = {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_job(job_id: str, **fields):
    conn = get_db()
    columns = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [job_id]
    conn.execute(f"UPDATE jobs SET {columns} WHERE id = ?", values)
    conn.commit()
    conn.close()


def get_job(job_id: str) -> Optional[sqlite3.Row]:
    conn = get_db()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    return row


async def run_mock_pipeline(job_id: str, user_query: str):
    """
    Simulates the agent -> manim -> tts -> ffmpeg pipeline.
    Replace this function body with a real call into the LangGraph
    orchestrator once it's ready; the job lifecycle and event shape
    below are what the frontend already expects, so nothing else
    needs to change.
    """
    try:
        update_job(job_id, status="running")
        await pubsub.publish(job_id, {"type": "status", "status": "running"})

        for stage_id, message in MOCK_STAGES:
            await asyncio.sleep(MOCK_STAGE_DELAY_SECONDS)
            await pubsub.publish(
                job_id,
                {"type": "progress", "stage": stage_id, "message": message},
            )

        final_video_path = str(MOCK_VIDEO_PATH)
        update_job(
            job_id,
            status="completed",
            final_video_path=final_video_path,
            completed_at=now(),
        )
        await pubsub.publish(
            job_id,
            {
                "type": "status",
                "status": "completed",
                "final_video_path": final_video_path,
            },
        )
    except asyncio.CancelledError:
        update_job(job_id, status="cancelled", completed_at=now())
        await pubsub.publish(job_id, {"type": "status", "status": "cancelled"})
        raise
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc), completed_at=now())
        await pubsub.publish(
            job_id, {"type": "status", "status": "failed", "error": str(exc)}
        )
    finally:
        running_tasks.pop(job_id, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="EduManim API", lifespan=lifespan)


class CreateJobRequest(BaseModel):
    user_query: str
    conversation_id: Optional[str] = None


class JobResponse(BaseModel):
    id: str
    conversation_id: Optional[str]
    user_query: str
    status: str
    final_video_path: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]


def row_to_response(row: sqlite3.Row) -> JobResponse:
    return JobResponse(
        id=row["id"],
        conversation_id=row["conversation_id"],
        user_query=row["user_query"],
        status=row["status"],
        final_video_path=row["final_video_path"],
        error=row["error"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )


@app.post("/api/jobs", response_model=JobResponse)
async def create_job(payload: CreateJobRequest):
    job_id = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        """
        INSERT INTO jobs (id, conversation_id, user_query, status, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (job_id, payload.conversation_id, payload.user_query, "queued", now()),
    )
    conn.commit()
    conn.close()

    task = asyncio.create_task(run_mock_pipeline(job_id, payload.user_query))
    running_tasks[job_id] = task

    return row_to_response(get_job(job_id))


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    row = get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return row_to_response(row)


@app.delete("/api/jobs/{job_id}", response_model=JobResponse)
async def cancel_job(job_id: str):
    row = get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if row["status"] not in ("queued", "running"):
        raise HTTPException(status_code=409, detail=f"Job already {row['status']}")

    task = running_tasks.get(job_id)
    if task:
        task.cancel()
    else:
        update_job(job_id, status="cancelled", completed_at=now())

    return row_to_response(get_job(job_id))


@app.websocket("/api/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str):
    if get_job(job_id) is None:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    queue = pubsub.subscribe(job_id)
    try:
        row = get_job(job_id)
        await websocket.send_text(
            json.dumps({"type": "status", "status": row["status"]})
        )
        while True:
            event = await queue.get()
            await websocket.send_text(json.dumps(event))
            if event.get("type") == "status" and event.get("status") in (
                "completed",
                "failed",
                "cancelled",
            ):
                break
    except WebSocketDisconnect:
        pass
    finally:
        pubsub.unsubscribe(job_id, queue)


@app.get("/api/videos/{job_id}")
async def stream_video(job_id: str):
    row = get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if row["status"] != "completed" or not row["final_video_path"]:
        raise HTTPException(status_code=409, detail="Video not ready yet")

    video_path = Path(row["final_video_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file missing on disk")

    return FileResponse(video_path, media_type="video/mp4", filename=video_path.name)


@app.get("/api/videos")
async def list_videos():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, final_video_path, completed_at FROM jobs WHERE status = 'completed'"
    ).fetchall()
    conn.close()
    return [
        {"job_id": r["id"], "video_path": r["final_video_path"], "completed_at": r["completed_at"]}
        for r in rows
    ]


@app.get("/health")
async def health():
    return {"status": "ok"}
