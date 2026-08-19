import json
import inspect
import math
import os
import random
import sys
import time
import uuid
from html import escape
from datetime import datetime, timedelta
from pathlib import Path
 
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
 

 
API_BASE_URL = os.environ.get("EDUMANIM_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_POLL_SECONDS = float(os.environ.get("EDUMANIM_API_POLL_SECONDS", "1.5"))
GENERATED_VIDEO_DIR = Path(os.environ.get("EDUMANIM_DATA_ROOT", "./data")) / "generated_videos"
GENERATED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
MAX_VIDEO_SLOTS = 15
 
# No system ffmpeg, no bundled .ttf files needed:
#  - imageio-ffmpeg ships its own ffmpeg binary as a pip package (installed
#    automatically as an `imageio` dependency), so there's nothing to
#    download separately or add to PATH.
#  - Pillow's ImageFont.load_default(size=...) is a built-in scalable font
#    (available since Pillow 9.2+), so no external font files are needed.
 
# Gradio's Chatbot API has shifted across versions: older releases (~4.20,
# your team's pin) accept messages-format history without a `type=` kwarg
# at all; mid-generation releases (~4.28-5.x) require `type="messages"`
# explicitly or default to the old tuple format; the newest releases (6.x)
# dropped `type=` entirely since messages format is now the only option.
# This keeps the app working across all of them instead of hardcoding one.
_CHATBOT_ACCEPTS_TYPE = "type" in inspect.signature(gr.Chatbot.__init__).parameters
CHATBOT_KWARGS = {"type": "messages"} if _CHATBOT_ACCEPTS_TYPE else {}
 
class ApiBackend:
    def __init__(self, base_url: str = API_BASE_URL, poll_seconds: float = API_POLL_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.poll_seconds = poll_seconds
        self.session = requests.Session()

    def create_job(self, query: str, voice: str, video_quality: str) -> dict:
        response = self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "user_query": query,
                "voice": voice,
                "video_quality": video_quality,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def get_job(self, job_id: str) -> dict:
        response = self.session.get(f"{self.base_url}/api/jobs/{job_id}", timeout=60)
        response.raise_for_status()
        return response.json()

    def stream_job(self, job_id: str):
        while True:
            job = self.get_job(job_id)
            yield job
            if job.get("status") in {"completed", "failed", "cancelled"}:
                break
            time.sleep(self.poll_seconds)

    def download_video(self, job_id: str) -> str:
        local_path = GENERATED_VIDEO_DIR / f"{job_id}.mp4"
        if local_path.exists():
            return str(local_path)

        with self.session.get(f"{self.base_url}/api/videos/{job_id}", stream=True, timeout=300) as response:
            response.raise_for_status()
            with local_path.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_handle.write(chunk)
        return str(local_path)

    def list_video_names(self) -> list[dict]:
        """Fetch only video filenames from the backend — no filesystem paths."""
        response = self.session.get(f"{self.base_url}/api/videos/names", timeout=10)
        response.raise_for_status()
        return response.json()

    def list_videos(self) -> list[dict]:
        response = self.session.get(f"{self.base_url}/api/videos", timeout=60)
        response.raise_for_status()
        return response.json()


_TITLE_FONT = ImageFont.load_default(size=28)
_SUBTITLE_FONT = ImageFont.load_default(size=15)

_LEADING_PHRASES = (
    "explain ", "what is ", "what are ", "how does ", "how do ",
    "why is ", "why does ", "tell me about ", "describe ",
)


def _extract_topic(query: str) -> str:
    cleaned = query.strip().rstrip("?.! ")
    lowered = cleaned.lower()
    for phrase in _LEADING_PHRASES:
        if lowered.startswith(phrase):
            cleaned = cleaned[len(phrase):]
            break
    return cleaned or "this topic"
 
 
def _fmt_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"
 
 
def format_transcript(scenes: list[dict]) -> str:
    """Scene-by-scene transcript rendered as clean HTML."""
    rows = []
    for i, scene in enumerate(scenes, start=1):
        rows.append(f"""
        <div class="em-transcript-scene">
          <div class="em-transcript-head">
            <span class="em-transcript-title">Scene {i} — {scene['title']}</span>
          </div>
          <p class="em-transcript-body">{scene['narration']}</p>
        </div>
        """)
    return f"<div class='em-transcript'>{''.join(rows)}</div>"


def build_transcript_from_job(job: dict) -> str:
    script_json = job.get("script_json")
    if not script_json:
        return "<div class='em-empty'>Transcript will appear here once a video is generated.</div>"

    try:
        script = json.loads(script_json)
    except json.JSONDecodeError:
        return "<div class='em-empty'>Transcript data was returned in an invalid format.</div>"

    scenes = script.get("scenes", []) if isinstance(script, dict) else []
    if not scenes:
        return "<div class='em-empty'>Transcript is not available for this job.</div>"
    return format_transcript(scenes)


def _has_transcript(job: dict) -> bool:
    script_json = job.get("script_json")
    if not script_json:
        return False
    try:
        script = json.loads(script_json)
    except json.JSONDecodeError:
        return False
    return bool(isinstance(script, dict) and script.get("scenes"))


def render_video_gallery(videos: list[dict]) -> str:
    if not videos:
        return "<div class='em-empty'>No videos yet — generate one from the Chat tab and it'll show up here.</div>"

    cards = []
    for video in videos:
        job_id = escape(str(video.get("job_id", "")))
        name = escape(str(video.get("name") or f"{job_id}.mp4"))
        completed_at = escape(str(video.get("completed_at", "")))
        video_url = f"{API_BASE_URL}/api/videos/{job_id}"
        safe_name = name.replace("'", "\\'").replace('"', '\\"')
        cards.append(
            f"""
            <div class='em-kb-row' style='display:flex; justify-content:space-between; gap:12px;'>
              <div>
                <div class='em-kb-label'><strong>{name}</strong></div>
                <div style='color: var(--em-muted); font-size: 12px;'>Completed: {completed_at}</div>
              </div>
              
              <!-- Un simplu tag <a> face toată magia direct din browser -->
              <a href="{video_url}" download="{safe_name}" target="_blank"
                 style='color:var(--em-teal); text-decoration:none; font-size:13px; align-self:center; font-weight:bold;'>
                ⬇ Download
              </a>
              
            </div>
            """
        )
    return "<div class='em-transcript'>" + "".join(cards) + "</div>"



backend = ApiBackend()
 
# ============================================================================
# THEME + CSS
# ============================================================================
 
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#0B1120",
    body_background_fill_dark="#0B1120",
    background_fill_primary="#121A2E",
    background_fill_secondary="#0F1729",
    border_color_primary="#22304F",
    body_text_color="#F5F3ED",
    body_text_color_subdued="#8B96AC",
    block_background_fill="#121A2E",
    block_border_color="#22304F",
    block_label_text_color="#8B96AC",
    button_primary_background_fill="#14B8A6",
    button_primary_background_fill_hover="#0FA394",
    button_primary_text_color="#0B1120",
    button_secondary_background_fill="#1B2740",
    button_secondary_text_color="#F5F3ED",
    input_background_fill="#0F1729",
    input_border_color="#22304F",
)
 
HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
"""
 
# The signature element: a coordinate axis + orbiting vector that draws and
# erases itself in a loop. This is the "hero as thesis" moment — it previews
# the exact visual language (Manim-style vector geometry) before any video
# has been generated.
SIGNATURE_SVG = """
<div class="em-sig-wrap">
  <svg viewBox="0 0 320 320" class="em-sig-svg">
    <circle cx="160" cy="160" r="110" class="em-orbit-ring" />
    <line x1="30" y1="160" x2="290" y2="160" class="em-axis" />
    <line x1="160" y1="30" x2="160" y2="290" class="em-axis" />
    <line x1="160" y1="160" x2="245" y2="95" class="em-vector" />
    <circle cx="245" cy="95" r="6" class="em-vector-tip" />
    <circle cx="160" cy="160" r="4" class="em-origin" />
    <text x="252" y="90" class="em-label">v</text>
  </svg>
</div>
"""
 
CUSTOM_CSS = """
:root {
  --em-void: #0B1120;
  --em-panel: #121A2E;
  --em-ink: #F5F3ED;
  --em-blue: #1E3A8A;
  --em-teal: #14B8A6;
  --em-muted: #8B96AC;
}
 
.gradio-container {
  background: var(--em-void) !important;
  max-width: 1180px !important;
  margin: 0 auto !important;
  width: 100% !important;
}
footer { display: none !important; }
 
h1, h2, h3, .em-display {
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: -0.01em;
}
 
/* ---------- Hero ---------- */
.em-hero {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 32px;
  padding: 40px 44px;
  margin-bottom: 8px;
  border-radius: 20px;
  background:
    radial-gradient(circle at 85% 20%, rgba(20,184,166,0.14), transparent 55%),
    radial-gradient(circle at 10% 90%, rgba(30,58,138,0.35), transparent 55%),
    linear-gradient(180deg, #0E1626 0%, #0B1120 100%);
  border: 1px solid #1C2740;
  overflow: hidden;
}
.em-hero::before {
  content: "";
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(245,243,237,0.028) 1px, transparent 1px),
    linear-gradient(90deg, rgba(245,243,237,0.028) 1px, transparent 1px);
  background-size: 34px 34px;
  mask-image: radial-gradient(ellipse 70% 60% at 30% 40%, black, transparent);
}
.em-hero-copy { position: relative; z-index: 1; max-width: 560px; }
.em-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12.5px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--em-teal); margin-bottom: 14px; display: block;
}
.em-hero h1 {
  font-size: 40px; line-height: 1.08; margin: 0 0 16px 0; color: var(--em-ink);
  font-weight: 700;
}
.em-hero h1 span { color: var(--em-teal); }
.em-hero p { color: var(--em-muted); font-size: 15.5px; line-height: 1.6; margin: 0; }
 
/* ---------- Signature animated diagram ---------- */
.em-sig-wrap { position: relative; z-index: 1; flex-shrink: 0; width: 220px; height: 220px; }
.em-sig-svg { width: 100%; height: 100%; overflow: visible; }
.em-axis { stroke: #26385E; stroke-width: 1.5; }
.em-orbit-ring {
  fill: none; stroke: #1C2740; stroke-width: 1;
}
.em-vector {
  fill: none; stroke: var(--em-teal); stroke-width: 3; stroke-linecap: round;
  stroke-dasharray: 130; stroke-dashoffset: 130;
  animation: em-draw 3.6s ease-in-out infinite;
  filter: drop-shadow(0 0 6px rgba(20,184,166,0.55));
}
.em-vector-tip { fill: var(--em-teal); opacity: 0; animation: em-tip 3.6s ease-in-out infinite; }
.em-origin { fill: var(--em-muted); }
.em-label { fill: var(--em-teal); font-family: 'JetBrains Mono', monospace; font-size: 15px; opacity: 0; animation: em-tip 3.6s ease-in-out infinite; }
 
@keyframes em-draw {
  0%   { stroke-dashoffset: 130; opacity: 0; }
  12%  { opacity: 1; }
  45%  { stroke-dashoffset: 0; opacity: 1; }
  70%  { stroke-dashoffset: 0; opacity: 1; }
  88%  { stroke-dashoffset: -130; opacity: 0.4; }
  100% { stroke-dashoffset: -130; opacity: 0; }
}
@keyframes em-tip {
  0%, 40% { opacity: 0; }
  45%, 70% { opacity: 1; }
  90%, 100% { opacity: 0; }
}
 
/* ---------- Agent thoughts terminal ---------- */
.em-terminal {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12.5px !important;
  background: #070C17 !important;
  border: 1px solid #1C2740 !important;
  border-radius: 10px !important;
  color: #7DD9CB !important;
  line-height: 1.7 !important;
}
 
/* ---------- Progress line ---------- */
.em-progress-label {
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  color: var(--em-teal); display: flex; align-items: center; gap: 8px;
}
.em-progress-label::before {
  content: ""; width: 7px; height: 7px; border-radius: 50%;
  background: var(--em-teal); animation: em-pulse 1.1s ease-in-out infinite;
  box-shadow: 0 0 8px rgba(20,184,166,0.9);
}
@keyframes em-pulse { 0%,100% { opacity: 0.35; } 50% { opacity: 1; } }
 
/* ---------- Tabs ---------- */
.tabs > .tab-nav { border-bottom: 1px solid #1C2740 !important; gap: 4px; }
.tabs > .tab-nav button {
  font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
  color: var(--em-muted) !important; border: none !important; background: transparent !important;
}
.tabs > .tab-nav button.selected {
  color: var(--em-teal) !important; border-bottom: 2px solid var(--em-teal) !important;
}
 
/* ---------- Empty states ---------- */
.em-empty {
  text-align: center; padding: 48px 20px; color: var(--em-muted);
  border: 1px dashed #22304F; border-radius: 14px; font-size: 14px;
}
 
/* ---------- KB document list ---------- */
.em-kb-row {
  border: 1px solid #1C2740 !important; border-radius: 10px !important;
  padding: 4px 14px !important; margin-bottom: 8px !important;
  background: #0F1729 !important; align-items: center !important;
}
.em-kb-row .em-kb-label { color: var(--em-ink) !important; font-size: 13.5px; }
.em-kb-row .em-kb-label strong { font-family: 'Space Grotesk', sans-serif; }
 
/* ---------- Transcript ---------- */
.em-transcript { display: flex; flex-direction: column; gap: 14px; margin-top: 6px; }
.em-transcript-scene {
  border-left: 2px solid #1C2740; padding: 2px 0 2px 14px;
}
.em-transcript-head {
  display: flex; justify-content: space-between; align-items: baseline; gap: 10px; margin-bottom: 4px;
}
.em-transcript-title {
  font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 13.5px; color: var(--em-teal);
}
.em-transcript-time {
  font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: var(--em-muted); flex-shrink: 0;
}
.em-transcript-body { color: var(--em-ink); font-size: 13.5px; line-height: 1.6; margin: 0; opacity: 0.92; }
"""
 
# ============================================================================
# CHAT TAB LOGIC
# ============================================================================
 
 
def handle_send(message, history, voice_state, quality_state):
    if not message or not message.strip():
        yield history, "", "", gr.update(visible=False), gr.update(visible=False)
        return
 
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "_starting…_"},
    ]
    yield history, "**Starting…**", "", gr.update(visible=False), gr.update(visible=False)
 
    try:
        job = backend.create_job(message, voice=voice_state, video_quality=quality_state)
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        history[-1]["content"] = f"⚠️ Could not start generation: {detail}"
        yield history, "<div class='em-progress-label'>failed</div>", detail, gr.update(visible=False), gr.update(visible=False)
        return
    except requests.RequestException as exc:
        history[-1]["content"] = f"⚠️ Could not connect to the API: {exc}"
        yield history, "<div class='em-progress-label'>failed</div>", str(exc), gr.update(visible=False), gr.update(visible=False)
        return

    thoughts = []
    transcript_html = gr.update(visible=False)
 
    try:
        for job_state in backend.stream_job(job["id"]):
            status = job_state.get("status", "running")
            stage = job_state.get("progress_stage") or status
            message_text = job_state.get("progress_message") or f"Job {status}"
            pct = int(_progress_fraction(job_state) * 100)
            status_md = f"<div class='em-progress-label'>{escape(stage)} · {pct}%</div>"

            if message_text and (not thoughts or thoughts[-1] != message_text):
                thoughts.append(message_text)

            if _has_transcript(job_state):
                transcript_html = gr.update(value=build_transcript_from_job(job_state), visible=True)

            history[-1]["content"] = f"_{message_text}_"
            yield history, status_md, "\n".join(thoughts), gr.update(visible=False), transcript_html

        final_job = backend.get_job(job["id"])
        if final_job.get("status") != "completed":
            error_text = final_job.get("error") or "Job did not complete successfully"
            history[-1]["content"] = f"⚠️ {error_text}"
            yield history, "<div class='em-progress-label'>failed · 100%</div>", "\n".join(thoughts), gr.update(visible=False), transcript_html
            return

        video_path = backend.download_video(job["id"])
        transcript_html = gr.update(value=build_transcript_from_job(final_job), visible=True)
        video_update = gr.update(value=video_path, visible=True)
        reply = "Here's your explainer video 🎬"

        history[-1]["content"] = reply
        yield (
            history,
            "<div class='em-progress-label'>done · 100%</div>",
            "\n".join(thoughts),
            video_update,
            transcript_html,
        )
        return
    except Exception as exc:  # noqa: BLE001 - keep the app alive even if the API flow fails
        history[-1]["content"] = f"⚠️ Generation failed: {exc}"
        yield history, "<div class='em-progress-label'>failed</div>", "\n".join(thoughts), gr.update(visible=False), transcript_html
        return


def _progress_fraction(job_state: dict) -> float:
    status = job_state.get("status")
    stage = job_state.get("progress_stage") or ""

    if status == "queued":
        return 0.05
    if status == "running":
        if stage.startswith("rendering_"):
            return 0.65
        if stage == "assembling":
            return 0.9
        if stage == "narration":
            return 0.45
        if stage == "scripting":
            return 0.2
        return 0.1
    if status == "completed":
        return 1.0
    if status in {"failed", "cancelled"}:
        return 1.0
    return 0.0

def refresh_video_gallery():
    try:
        videos = backend.list_video_names()
    except Exception as exc:
        videos = []
        print(f"Eroare la încărcarea videoclipurilor: {exc}")

    updates = []
    updates.append(gr.update(visible=len(videos) == 0))

    for i in range(MAX_VIDEO_SLOTS):
        if i < len(videos):
            video = videos[i]
            job_id = video.get("job_id", "")
            name = escape(str(video.get("name") or f"{job_id}.mp4"))
            completed_at = escape(str(video.get("completed_at", "")))
            
            local_path = str(GENERATED_VIDEO_DIR / f"{job_id}.mp4")
            
            html_label = f"""
                <div class='em-kb-label'><strong>{name}</strong></div>
                <div style='color: var(--em-muted); font-size: 12px;'>Completed: {completed_at}</div>
            """
            
            updates.append(gr.update(visible=True))
            updates.append(gr.update(value=html_label))
            updates.append(gr.update(value=local_path))
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=""))
            updates.append(gr.update(value=None))

    return updates
 

 
 
# ============================================================================
# BUILD APP
# ============================================================================
 
def build_app() -> gr.Blocks:
    with gr.Blocks(theme=THEME, css=CUSTOM_CSS, head=HEAD_HTML, title="EduManim") as demo:
 
        voice_state = gr.State(value="Narrator")
        quality_state = gr.State(value="720p")
 
        # ---------------- Hero ----------------
        gr.HTML(f"""
        <div class="em-hero">
          <div class="em-hero-copy">
            <span class="em-eyebrow">Local · Private · AMD Radeon</span>
            <h1>Ask something.<br/>Watch it get <span>explained</span>.</h1>
            <p>EduManim turns a question into a narrated, animated explainer video —
            planned, researched, rendered, and voiced entirely on your own GPU.</p>
          </div>
          {SIGNATURE_SVG}
        </div>
        """)
 
        with gr.Tabs():
 
            # ============================= CHAT =============================
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            height=380,
                            label=None,
                            show_label=False,
                            avatar_images=(None, None),
                            **CHATBOT_KWARGS,
                        )
 
                        with gr.Row():
                            msg_box = gr.Textbox(
                                placeholder="Explain self-attention…",
                                show_label=False,
                                scale=5,
                                container=False,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
 
                    with gr.Column(scale=2):
                        progress_md = gr.HTML("<div class='em-progress-label' style='opacity:0.4'>waiting for a question…</div>")
                        video_player = gr.Video(visible=False, show_label=False, height=220)
                        with gr.Accordion("Transcript", open=True):
                            transcript_html = gr.HTML("<div class='em-empty'>Transcript will appear here once a video is generated.</div>")
                        with gr.Accordion("Agent thoughts", open=False):
                            thoughts_box = gr.Textbox(
                                value="",
                                lines=10,
                                show_label=False,
                                interactive=False,
                                elem_classes=["em-terminal"],
                            )
 
            # ============================ MY VIDEOS ==========================
            with gr.Tab("🎬 My Videos") as my_videos_tab:
                videos_empty = gr.HTML("<div class='em-empty'>Click this tab to load your videos.</div>")
                
                video_rows = []
                video_labels = []
                video_dl_btns = []

                for i in range(MAX_VIDEO_SLOTS):
                    with gr.Row(visible=False, elem_classes=["em-kb-row"]) as row:
                        lbl = gr.HTML()
                        btn = gr.DownloadButton("⬇ Download", size="sm")
                        
                        video_rows.append(row)
                        video_labels.append(lbl)
                        video_dl_btns.append(btn)

            # ============================ SETTINGS ===========================
            with gr.Tab("⚙️ Settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Voice**")
                        voice_radio = gr.Radio(
                            ["Narrator", "Female", "Male"],
                            value="Narrator",
                            show_label=False,
                        )
                        gr.Markdown("**Video quality**")
                        quality_radio = gr.Radio(["720p", "1080p"], value="720p", show_label=False)
 
                voice_radio.change(lambda v: v, [voice_radio], [voice_state])
                quality_radio.change(lambda v: v, [quality_radio], [quality_state])
 
            video_gallery_outputs = [videos_empty]
            for r, l, b in zip(video_rows, video_labels, video_dl_btns):
                video_gallery_outputs.extend([r, l, b])

            send_btn.click(
                handle_send,
                [msg_box, chatbot, voice_state, quality_state],
                [chatbot, progress_md, thoughts_box, video_player, transcript_html],
            ).then(refresh_video_gallery, None, video_gallery_outputs).then(lambda: "", None, msg_box)

            msg_box.submit(
                handle_send,
                [msg_box, chatbot, voice_state, quality_state],
                [chatbot, progress_md, thoughts_box, video_player, transcript_html],
            ).then(refresh_video_gallery, None, video_gallery_outputs).then(lambda: "", None, msg_box)
            
            my_videos_tab.select(refresh_video_gallery, None, video_gallery_outputs)
    
    return demo
 
 
if __name__ == "__main__":
    app = build_app()
    app.queue().launch(server_name="0.0.0.0", server_port=7860) import json
import inspect
import math
import os
import random
import sys
import time
import uuid
from html import escape
from datetime import datetime, timedelta
from pathlib import Path
 
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
 

 
API_BASE_URL = os.environ.get("EDUMANIM_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_POLL_SECONDS = float(os.environ.get("EDUMANIM_API_POLL_SECONDS", "1.5"))
GENERATED_VIDEO_DIR = Path(os.environ.get("EDUMANIM_DATA_ROOT", "./data")) / "generated_videos"
GENERATED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
MAX_VIDEO_SLOTS = 15
 
# No system ffmpeg, no bundled .ttf files needed:
#  - imageio-ffmpeg ships its own ffmpeg binary as a pip package (installed
#    automatically as an `imageio` dependency), so there's nothing to
#    download separately or add to PATH.
#  - Pillow's ImageFont.load_default(size=...) is a built-in scalable font
#    (available since Pillow 9.2+), so no external font files are needed.
 
# Gradio's Chatbot API has shifted across versions: older releases (~4.20,
# your team's pin) accept messages-format history without a `type=` kwarg
# at all; mid-generation releases (~4.28-5.x) require `type="messages"`
# explicitly or default to the old tuple format; the newest releases (6.x)
# dropped `type=` entirely since messages format is now the only option.
# This keeps the app working across all of them instead of hardcoding one.
_CHATBOT_ACCEPTS_TYPE = "type" in inspect.signature(gr.Chatbot.__init__).parameters
CHATBOT_KWARGS = {"type": "messages"} if _CHATBOT_ACCEPTS_TYPE else {}
 
class ApiBackend:
    def __init__(self, base_url: str = API_BASE_URL, poll_seconds: float = API_POLL_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.poll_seconds = poll_seconds
        self.session = requests.Session()

    def create_job(self, query: str, voice: str, video_quality: str) -> dict:
        response = self.session.post(
            f"{self.base_url}/api/jobs",
            json={
                "user_query": query,
                "voice": voice,
                "video_quality": video_quality,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def get_job(self, job_id: str) -> dict:
        response = self.session.get(f"{self.base_url}/api/jobs/{job_id}", timeout=60)
        response.raise_for_status()
        return response.json()

    def stream_job(self, job_id: str):
        while True:
            job = self.get_job(job_id)
            yield job
            if job.get("status") in {"completed", "failed", "cancelled"}:
                break
            time.sleep(self.poll_seconds)

    def download_video(self, job_id: str) -> str:
        local_path = GENERATED_VIDEO_DIR / f"{job_id}.mp4"
        if local_path.exists():
            return str(local_path)

        with self.session.get(f"{self.base_url}/api/videos/{job_id}", stream=True, timeout=300) as response:
            response.raise_for_status()
            with local_path.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_handle.write(chunk)
        return str(local_path)

    def list_video_names(self) -> list[dict]:
        """Fetch only video filenames from the backend — no filesystem paths."""
        response = self.session.get(f"{self.base_url}/api/videos/names", timeout=10)
        response.raise_for_status()
        return response.json()

    def list_videos(self) -> list[dict]:
        response = self.session.get(f"{self.base_url}/api/videos", timeout=60)
        response.raise_for_status()
        return response.json()


_TITLE_FONT = ImageFont.load_default(size=28)
_SUBTITLE_FONT = ImageFont.load_default(size=15)

_LEADING_PHRASES = (
    "explain ", "what is ", "what are ", "how does ", "how do ",
    "why is ", "why does ", "tell me about ", "describe ",
)


def _extract_topic(query: str) -> str:
    cleaned = query.strip().rstrip("?.! ")
    lowered = cleaned.lower()
    for phrase in _LEADING_PHRASES:
        if lowered.startswith(phrase):
            cleaned = cleaned[len(phrase):]
            break
    return cleaned or "this topic"
 
 
def _fmt_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"
 
 
def format_transcript(scenes: list[dict]) -> str:
    """Scene-by-scene transcript rendered as clean HTML."""
    rows = []
    for i, scene in enumerate(scenes, start=1):
        rows.append(f"""
        <div class="em-transcript-scene">
          <div class="em-transcript-head">
            <span class="em-transcript-title">Scene {i} — {scene['title']}</span>
          </div>
          <p class="em-transcript-body">{scene['narration']}</p>
        </div>
        """)
    return f"<div class='em-transcript'>{''.join(rows)}</div>"


def build_transcript_from_job(job: dict) -> str:
    script_json = job.get("script_json")
    if not script_json:
        return "<div class='em-empty'>Transcript will appear here once a video is generated.</div>"

    try:
        script = json.loads(script_json)
    except json.JSONDecodeError:
        return "<div class='em-empty'>Transcript data was returned in an invalid format.</div>"

    scenes = script.get("scenes", []) if isinstance(script, dict) else []
    if not scenes:
        return "<div class='em-empty'>Transcript is not available for this job.</div>"
    return format_transcript(scenes)


def _has_transcript(job: dict) -> bool:
    script_json = job.get("script_json")
    if not script_json:
        return False
    try:
        script = json.loads(script_json)
    except json.JSONDecodeError:
        return False
    return bool(isinstance(script, dict) and script.get("scenes"))


def render_video_gallery(videos: list[dict]) -> str:
    if not videos:
        return "<div class='em-empty'>No videos yet — generate one from the Chat tab and it'll show up here.</div>"

    cards = []
    for video in videos:
        job_id = escape(str(video.get("job_id", "")))
        name = escape(str(video.get("name") or f"{job_id}.mp4"))
        completed_at = escape(str(video.get("completed_at", "")))
        video_url = f"{API_BASE_URL}/api/videos/{job_id}"
        safe_name = name.replace("'", "\\'").replace('"', '\\"')
        cards.append(
            f"""
            <div class='em-kb-row' style='display:flex; justify-content:space-between; gap:12px;'>
              <div>
                <div class='em-kb-label'><strong>{name}</strong></div>
                <div style='color: var(--em-muted); font-size: 12px;'>Completed: {completed_at}</div>
              </div>
              
              <!-- Un simplu tag <a> face toată magia direct din browser -->
              <a href="{video_url}" download="{safe_name}" target="_blank"
                 style='color:var(--em-teal); text-decoration:none; font-size:13px; align-self:center; font-weight:bold;'>
                ⬇ Download
              </a>
              
            </div>
            """
        )
    return "<div class='em-transcript'>" + "".join(cards) + "</div>"



backend = ApiBackend()
 
# ============================================================================
# THEME + CSS
# ============================================================================
 
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#0B1120",
    body_background_fill_dark="#0B1120",
    background_fill_primary="#121A2E",
    background_fill_secondary="#0F1729",
    border_color_primary="#22304F",
    body_text_color="#F5F3ED",
    body_text_color_subdued="#8B96AC",
    block_background_fill="#121A2E",
    block_border_color="#22304F",
    block_label_text_color="#8B96AC",
    button_primary_background_fill="#14B8A6",
    button_primary_background_fill_hover="#0FA394",
    button_primary_text_color="#0B1120",
    button_secondary_background_fill="#1B2740",
    button_secondary_text_color="#F5F3ED",
    input_background_fill="#0F1729",
    input_border_color="#22304F",
)
 
HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
"""
 
# The signature element: a coordinate axis + orbiting vector that draws and
# erases itself in a loop. This is the "hero as thesis" moment — it previews
# the exact visual language (Manim-style vector geometry) before any video
# has been generated.
SIGNATURE_SVG = """
<div class="em-sig-wrap">
  <svg viewBox="0 0 320 320" class="em-sig-svg">
    <circle cx="160" cy="160" r="110" class="em-orbit-ring" />
    <line x1="30" y1="160" x2="290" y2="160" class="em-axis" />
    <line x1="160" y1="30" x2="160" y2="290" class="em-axis" />
    <line x1="160" y1="160" x2="245" y2="95" class="em-vector" />
    <circle cx="245" cy="95" r="6" class="em-vector-tip" />
    <circle cx="160" cy="160" r="4" class="em-origin" />
    <text x="252" y="90" class="em-label">v</text>
  </svg>
</div>
"""
 
CUSTOM_CSS = """
:root {
  --em-void: #0B1120;
  --em-panel: #121A2E;
  --em-ink: #F5F3ED;
  --em-blue: #1E3A8A;
  --em-teal: #14B8A6;
  --em-muted: #8B96AC;
}
 
.gradio-container {
  background: var(--em-void) !important;
  max-width: 1180px !important;
  margin: 0 auto !important;
  width: 100% !important;
}
footer { display: none !important; }
 
h1, h2, h3, .em-display {
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: -0.01em;
}
 
/* ---------- Hero ---------- */
.em-hero {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 32px;
  padding: 40px 44px;
  margin-bottom: 8px;
  border-radius: 20px;
  background:
    radial-gradient(circle at 85% 20%, rgba(20,184,166,0.14), transparent 55%),
    radial-gradient(circle at 10% 90%, rgba(30,58,138,0.35), transparent 55%),
    linear-gradient(180deg, #0E1626 0%, #0B1120 100%);
  border: 1px solid #1C2740;
  overflow: hidden;
}
.em-hero::before {
  content: "";
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(245,243,237,0.028) 1px, transparent 1px),
    linear-gradient(90deg, rgba(245,243,237,0.028) 1px, transparent 1px);
  background-size: 34px 34px;
  mask-image: radial-gradient(ellipse 70% 60% at 30% 40%, black, transparent);
}
.em-hero-copy { position: relative; z-index: 1; max-width: 560px; }
.em-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12.5px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--em-teal); margin-bottom: 14px; display: block;
}
.em-hero h1 {
  font-size: 40px; line-height: 1.08; margin: 0 0 16px 0; color: var(--em-ink);
  font-weight: 700;
}
.em-hero h1 span { color: var(--em-teal); }
.em-hero p { color: var(--em-muted); font-size: 15.5px; line-height: 1.6; margin: 0; }
 
/* ---------- Signature animated diagram ---------- */
.em-sig-wrap { position: relative; z-index: 1; flex-shrink: 0; width: 220px; height: 220px; }
.em-sig-svg { width: 100%; height: 100%; overflow: visible; }
.em-axis { stroke: #26385E; stroke-width: 1.5; }
.em-orbit-ring {
  fill: none; stroke: #1C2740; stroke-width: 1;
}
.em-vector {
  fill: none; stroke: var(--em-teal); stroke-width: 3; stroke-linecap: round;
  stroke-dasharray: 130; stroke-dashoffset: 130;
  animation: em-draw 3.6s ease-in-out infinite;
  filter: drop-shadow(0 0 6px rgba(20,184,166,0.55));
}
.em-vector-tip { fill: var(--em-teal); opacity: 0; animation: em-tip 3.6s ease-in-out infinite; }
.em-origin { fill: var(--em-muted); }
.em-label { fill: var(--em-teal); font-family: 'JetBrains Mono', monospace; font-size: 15px; opacity: 0; animation: em-tip 3.6s ease-in-out infinite; }
 
@keyframes em-draw {
  0%   { stroke-dashoffset: 130; opacity: 0; }
  12%  { opacity: 1; }
  45%  { stroke-dashoffset: 0; opacity: 1; }
  70%  { stroke-dashoffset: 0; opacity: 1; }
  88%  { stroke-dashoffset: -130; opacity: 0.4; }
  100% { stroke-dashoffset: -130; opacity: 0; }
}
@keyframes em-tip {
  0%, 40% { opacity: 0; }
  45%, 70% { opacity: 1; }
  90%, 100% { opacity: 0; }
}
 
/* ---------- Agent thoughts terminal ---------- */
.em-terminal {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12.5px !important;
  background: #070C17 !important;
  border: 1px solid #1C2740 !important;
  border-radius: 10px !important;
  color: #7DD9CB !important;
  line-height: 1.7 !important;
}
 
/* ---------- Progress line ---------- */
.em-progress-label {
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  color: var(--em-teal); display: flex; align-items: center; gap: 8px;
}
.em-progress-label::before {
  content: ""; width: 7px; height: 7px; border-radius: 50%;
  background: var(--em-teal); animation: em-pulse 1.1s ease-in-out infinite;
  box-shadow: 0 0 8px rgba(20,184,166,0.9);
}
@keyframes em-pulse { 0%,100% { opacity: 0.35; } 50% { opacity: 1; } }
 
/* ---------- Tabs ---------- */
.tabs > .tab-nav { border-bottom: 1px solid #1C2740 !important; gap: 4px; }
.tabs > .tab-nav button {
  font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
  color: var(--em-muted) !important; border: none !important; background: transparent !important;
}
.tabs > .tab-nav button.selected {
  color: var(--em-teal) !important; border-bottom: 2px solid var(--em-teal) !important;
}
 
/* ---------- Empty states ---------- */
.em-empty {
  text-align: center; padding: 48px 20px; color: var(--em-muted);
  border: 1px dashed #22304F; border-radius: 14px; font-size: 14px;
}
 
/* ---------- KB document list ---------- */
.em-kb-row {
  border: 1px solid #1C2740 !important; border-radius: 10px !important;
  padding: 4px 14px !important; margin-bottom: 8px !important;
  background: #0F1729 !important; align-items: center !important;
}
.em-kb-row .em-kb-label { color: var(--em-ink) !important; font-size: 13.5px; }
.em-kb-row .em-kb-label strong { font-family: 'Space Grotesk', sans-serif; }
 
/* ---------- Transcript ---------- */
.em-transcript { display: flex; flex-direction: column; gap: 14px; margin-top: 6px; }
.em-transcript-scene {
  border-left: 2px solid #1C2740; padding: 2px 0 2px 14px;
}
.em-transcript-head {
  display: flex; justify-content: space-between; align-items: baseline; gap: 10px; margin-bottom: 4px;
}
.em-transcript-title {
  font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 13.5px; color: var(--em-teal);
}
.em-transcript-time {
  font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: var(--em-muted); flex-shrink: 0;
}
.em-transcript-body { color: var(--em-ink); font-size: 13.5px; line-height: 1.6; margin: 0; opacity: 0.92; }
"""
 
# ============================================================================
# CHAT TAB LOGIC
# ============================================================================
 
 
def handle_send(message, history, voice_state, quality_state):
    if not message or not message.strip():
        yield history, "", "", gr.update(visible=False), gr.update(visible=False)
        return
 
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "_starting…_"},
    ]
    yield history, "**Starting…**", "", gr.update(visible=False), gr.update(visible=False)
 
    try:
        job = backend.create_job(message, voice=voice_state, video_quality=quality_state)
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        history[-1]["content"] = f"⚠️ Could not start generation: {detail}"
        yield history, "<div class='em-progress-label'>failed</div>", detail, gr.update(visible=False), gr.update(visible=False)
        return
    except requests.RequestException as exc:
        history[-1]["content"] = f"⚠️ Could not connect to the API: {exc}"
        yield history, "<div class='em-progress-label'>failed</div>", str(exc), gr.update(visible=False), gr.update(visible=False)
        return

    thoughts = []
    transcript_html = gr.update(visible=False)
 
    try:
        for job_state in backend.stream_job(job["id"]):
            status = job_state.get("status", "running")
            stage = job_state.get("progress_stage") or status
            message_text = job_state.get("progress_message") or f"Job {status}"
            pct = int(_progress_fraction(job_state) * 100)
            status_md = f"<div class='em-progress-label'>{escape(stage)} · {pct}%</div>"

            if message_text and (not thoughts or thoughts[-1] != message_text):
                thoughts.append(message_text)

            if _has_transcript(job_state):
                transcript_html = gr.update(value=build_transcript_from_job(job_state), visible=True)

            history[-1]["content"] = f"_{message_text}_"
            yield history, status_md, "\n".join(thoughts), gr.update(visible=False), transcript_html

        final_job = backend.get_job(job["id"])
        if final_job.get("status") != "completed":
            error_text = final_job.get("error") or "Job did not complete successfully"
            history[-1]["content"] = f"⚠️ {error_text}"
            yield history, "<div class='em-progress-label'>failed · 100%</div>", "\n".join(thoughts), gr.update(visible=False), transcript_html
            return

        video_path = backend.download_video(job["id"])
        transcript_html = gr.update(value=build_transcript_from_job(final_job), visible=True)
        video_update = gr.update(value=video_path, visible=True)
        reply = "Here's your explainer video 🎬"

        history[-1]["content"] = reply
        yield (
            history,
            "<div class='em-progress-label'>done · 100%</div>",
            "\n".join(thoughts),
            video_update,
            transcript_html,
        )
        return
    except Exception as exc:  # noqa: BLE001 - keep the app alive even if the API flow fails
        history[-1]["content"] = f"⚠️ Generation failed: {exc}"
        yield history, "<div class='em-progress-label'>failed</div>", "\n".join(thoughts), gr.update(visible=False), transcript_html
        return


def _progress_fraction(job_state: dict) -> float:
    status = job_state.get("status")
    stage = job_state.get("progress_stage") or ""

    if status == "queued":
        return 0.05
    if status == "running":
        if stage.startswith("rendering_"):
            return 0.65
        if stage == "assembling":
            return 0.9
        if stage == "narration":
            return 0.45
        if stage == "scripting":
            return 0.2
        return 0.1
    if status == "completed":
        return 1.0
    if status in {"failed", "cancelled"}:
        return 1.0
    return 0.0

def refresh_video_gallery():
    try:
        videos = backend.list_video_names()
    except Exception as exc:
        videos = []
        print(f"Eroare la încărcarea videoclipurilor: {exc}")

    updates = []
    updates.append(gr.update(visible=len(videos) == 0))

    for i in range(MAX_VIDEO_SLOTS):
        if i < len(videos):
            video = videos[i]
            job_id = video.get("job_id", "")
            name = escape(str(video.get("name") or f"{job_id}.mp4"))
            completed_at = escape(str(video.get("completed_at", "")))
            
            local_path = str(GENERATED_VIDEO_DIR / f"{job_id}.mp4")
            
            html_label = f"""
                <div class='em-kb-label'><strong>{name}</strong></div>
                <div style='color: var(--em-muted); font-size: 12px;'>Completed: {completed_at}</div>
            """
            
            updates.append(gr.update(visible=True))
            updates.append(gr.update(value=html_label))
            updates.append(gr.update(value=local_path))
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=""))
            updates.append(gr.update(value=None))

    return updates
 

 
 
# ============================================================================
# BUILD APP
# ============================================================================
 
def build_app() -> gr.Blocks:
    with gr.Blocks(theme=THEME, css=CUSTOM_CSS, head=HEAD_HTML, title="EduManim") as demo:
 
        voice_state = gr.State(value="Narrator")
        quality_state = gr.State(value="720p")
 
        # ---------------- Hero ----------------
        gr.HTML(f"""
        <div class="em-hero">
          <div class="em-hero-copy">
            <span class="em-eyebrow">Local · Private · AMD Radeon</span>
            <h1>Ask something.<br/>Watch it get <span>explained</span>.</h1>
            <p>EduManim turns a question into a narrated, animated explainer video —
            planned, researched, rendered, and voiced entirely on your own GPU.</p>
          </div>
          {SIGNATURE_SVG}
        </div>
        """)
 
        with gr.Tabs():
 
            # ============================= CHAT =============================
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            height=380,
                            label=None,
                            show_label=False,
                            avatar_images=(None, None),
                            **CHATBOT_KWARGS,
                        )
 
                        with gr.Row():
                            msg_box = gr.Textbox(
                                placeholder="Explain self-attention…",
                                show_label=False,
                                scale=5,
                                container=False,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
 
                    with gr.Column(scale=2):
                        progress_md = gr.HTML("<div class='em-progress-label' style='opacity:0.4'>waiting for a question…</div>")
                        video_player = gr.Video(visible=False, show_label=False, height=220)
                        with gr.Accordion("Transcript", open=True):
                            transcript_html = gr.HTML("<div class='em-empty'>Transcript will appear here once a video is generated.</div>")
                        with gr.Accordion("Agent thoughts", open=False):
                            thoughts_box = gr.Textbox(
                                value="",
                                lines=10,
                                show_label=False,
                                interactive=False,
                                elem_classes=["em-terminal"],
                            )
 
            # ============================ MY VIDEOS ==========================
            with gr.Tab("🎬 My Videos") as my_videos_tab:
                videos_empty = gr.HTML("<div class='em-empty'>Click this tab to load your videos.</div>")
                
                video_rows = []
                video_labels = []
                video_dl_btns = []

                for i in range(MAX_VIDEO_SLOTS):
                    with gr.Row(visible=False, elem_classes=["em-kb-row"]) as row:
                        lbl = gr.HTML()
                        btn = gr.DownloadButton("⬇ Download", size="sm")
                        
                        video_rows.append(row)
                        video_labels.append(lbl)
                        video_dl_btns.append(btn)

            # ============================ SETTINGS ===========================
            with gr.Tab("⚙️ Settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Voice**")
                        voice_radio = gr.Radio(
                            ["Narrator", "Female", "Male"],
                            value="Narrator",
                            show_label=False,
                        )
                        gr.Markdown("**Video quality**")
                        quality_radio = gr.Radio(["720p", "1080p"], value="720p", show_label=False)
 
                voice_radio.change(lambda v: v, [voice_radio], [voice_state])
                quality_radio.change(lambda v: v, [quality_radio], [quality_state])
 
            video_gallery_outputs = [videos_empty]
            for r, l, b in zip(video_rows, video_labels, video_dl_btns):
                video_gallery_outputs.extend([r, l, b])

            send_btn.click(
                handle_send,
                [msg_box, chatbot, voice_state, quality_state],
                [chatbot, progress_md, thoughts_box, video_player, transcript_html],
            ).then(refresh_video_gallery, None, video_gallery_outputs).then(lambda: "", None, msg_box)

            msg_box.submit(
                handle_send,
                [msg_box, chatbot, voice_state, quality_state],
                [chatbot, progress_md, thoughts_box, video_player, transcript_html],
            ).then(refresh_video_gallery, None, video_gallery_outputs).then(lambda: "", None, msg_box)
            
            my_videos_tab.select(refresh_video_gallery, None, video_gallery_outputs)
    
    return demo
 
 
if __name__ == "__main__":
    app = build_app()
    app.queue().launch(server_name="0.0.0.0", server_port=7860) 