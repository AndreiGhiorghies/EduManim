
import inspect
import math
import os
import random
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
 
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
 
# frontend/app.py needs to reach the sibling tools/rag/ package:
#   edumanim/
#   ├── tools/rag/...
#   └── frontend/app.py   <- this file
# Running `python frontend/app.py` puts frontend/ on sys.path, not the
# project root, so tools/ wouldn't be importable without this.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
try:
    from tools.rag.config import DEFAULT_KB_ID
    from tools.rag.ingest import delete_document, ingest_file, list_documents, reindex_all
    from tools.rag.retrieve import Retriever
 
    RAG_AVAILABLE = True
except ImportError as _rag_import_error:
    RAG_AVAILABLE = False
    DEFAULT_KB_ID = "default"
    _RAG_IMPORT_ERROR_MSG = (
        f"RAG module not found at {_PROJECT_ROOT / 'tools' / 'rag'} "
        f"({_rag_import_error}). Knowledge Base features are disabled "
        f"until tools/rag/ is present with its dependencies installed."
    )
 
MOCK_VIDEO_DIR = Path(os.environ.get("EDUMANIM_DATA_ROOT", "./data")) / "mock_videos"
MOCK_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
 
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
 
# ============================================================================
# MOCK BACKEND
# Same shape Track A's real API will have: create_job / stream_progress /
# list_kb_docs / upload_doc / delete_doc / list_videos. Swap the class, not
# the calls in the UI functions below.
# ============================================================================
 
PROGRESS_STEPS = [
    ("planning", "Planning the explanation", 0.10),
    ("researching", "Researching your knowledge base + the web", 0.28),
    ("scripting", "Writing the narration script", 0.45),
    ("rendering", "Rendering scene 1 of 3", 0.60),
    ("rendering", "Rendering scene 2 of 3", 0.74),
    ("rendering", "Rendering scene 3 of 3", 0.85),
    ("narrating", "Generating voiceover", 0.93),
    ("assembling", "Assembling the final video", 1.00),
]
 
THOUGHT_LINES = {
    "planning": "→ plan: [research, outline(4 scenes), render, narrate, assemble]",
    "researching": "→ kb_search('{q}')  |  web_search('{q} explained')",
    "scripting": "→ script.scenes = 3, target runtime ≈ 2m10s",
    "rendering": "→ manim -qh scene_{n}.py Scene{n} --media_dir ./data/manim_workspace",
    "narrating": "→ tts.synthesize(narration, voice='{voice}')",
    "assembling": "→ ffmpeg: mux audio → concat scenes → final.mp4",
}
 
 
class MockJob:
    def __init__(self, query, kb_id=None, voice="Narrator (neutral)"):
        self.job_id = str(uuid.uuid4())[:8]
        self.query = query
        self.kb_id = kb_id
        self.voice = voice
        self.created_at = datetime.now()
 
 
class MockBackend:
    """Standalone dev backend for job/video/transcript mocking (still no
    Track A, no LLM, no Manim). Knowledge Base storage/retrieval is NOT
    mocked -- it calls the real tools/rag/ module directly, see
    handle_upload/handle_slot_delete/handle_test_query below.
    """
 
    def __init__(self):
        self.jobs: dict[str, MockJob] = {}
 
    def create_job(self, query: str, kb_id: str | None = None, voice: str = "Narrator (neutral)") -> MockJob:
        job = MockJob(query, kb_id, voice)
        self.jobs[job.job_id] = job
        return job
 
    def stream_progress(self, job: MockJob):
        """Yields (status_line, thought_line, progress_fraction) tuples.
        The 'researching' step calls your REAL RAG retriever if a
        knowledge base has documents in it -- everything else here
        (planning/scripting/rendering/narrating) is still simulated since
        it depends on the LLM + Manim + TTS, which are Track A/B's job.
        """
        for status, label, frac in PROGRESS_STEPS:
            time.sleep(random.uniform(0.35, 0.7))
            thought = THOUGHT_LINES.get(status, "")
            if status == "researching" and RAG_AVAILABLE:
                thought = _real_research_thought(job)
            else:
                if "{q}" in thought:
                    thought = thought.format(q=job.query[:40])
                if "{n}" in thought:
                    n = label.split(" ")[2] if "scene" in label else "1"
                    thought = thought.format(n=n)
                if "{voice}" in thought:
                    thought = thought.format(voice=job.voice)
            yield label, thought, frac
 
    def build_scenes(self, query: str) -> list[dict]:
        """Mock 'script' -- what Track A's Scriptwriter node would actually
        return. Generic on purpose since this is standalone dev, not a real
        LLM call.
        """
        topic = _extract_topic(query)
        return [
            {
                "title": "Introduction",
                "narration": f"Let's break down {topic}. Before the details, here's the "
                              f"big picture of what's actually going on and why it matters.",
            },
            {
                "title": "Core idea",
                "narration": f"At the heart of {topic} is a simple mechanism, built step "
                              f"by step from a few key pieces working together.",
            },
            {
                "title": "Worked example",
                "narration": f"Let's walk through a concrete example of {topic}, so the "
                              f"idea isn't just abstract -- you can see exactly how it plays out.",
            },
            {
                "title": "Summary",
                "narration": f"To recap: {topic} comes down to the core idea we just "
                              f"covered. Keep that mental model in mind next time you run into it.",
            },
        ]
 
    def render_preview_video(self, job: MockJob) -> str:
        """Synthesizes a short, REAL, playable mp4 -- pure Python (Pillow
        draws frames, imageio-ffmpeg encodes them), no system ffmpeg
        install or font files required. Not the real rendered Manim scenes
        (that's Track A + B's job post-integration) -- clearly labeled as
        a preview so it's never mistaken for the final render.
        """
        import imageio.v2 as imageio
 
        out_path = MOCK_VIDEO_DIR / f"{job.job_id}.mp4"
        width, height, fps, duration = 960, 544, 30, 5
        title = job.query[:60]
        subtitle = f"EduManim preview  ·  voiced by {job.voice}"
 
        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=6)
        try:
            for i in range(fps * duration):
                frame = _render_frame(width, height, i / fps, title, subtitle)
                writer.append_data(frame)
        finally:
            writer.close()
 
        return str(out_path)
 
 
_TITLE_FONT = ImageFont.load_default(size=28)
_SUBTITLE_FONT = ImageFont.load_default(size=15)
 
_BG = (11, 17, 32)       # #0B1120
_INK = (245, 243, 237)   # #F5F3ED
_TEAL = (20, 184, 166)   # #14B8A6
 
 
def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
 
 
def _render_frame(width: int, height: int, t: float, title: str, subtitle: str) -> np.ndarray:
    """One frame of the mock preview: dark background, a pulsing teal ring
    (echoing the hero's animated vector diagram), title + subtitle text.
    Pure Pillow -- no external font files, no ffmpeg CLI calls.
    """
    img = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(img)
 
    # Faint grid, matching the hero's background texture
    for x in range(0, width, 34):
        draw.line([(x, 0), (x, height)], fill=(18, 26, 46), width=1)
    for y in range(0, height, 34):
        draw.line([(0, y), (width, y)], fill=(18, 26, 46), width=1)
 
    # Pulsing ring + orbiting dot, standing in for the real rendered scene
    cx, cy = width // 2, int(height * 0.52)
    r = 70 + 14 * math.sin(t * 2.2)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=_TEAL, width=3)
    angle = t * 1.6
    ox, oy = cx + r * math.cos(angle), cy + r * math.sin(angle)
    draw.ellipse([ox - 7, oy - 7, ox + 7, oy + 7], fill=_TEAL)
 
    # Title (wrapped) + subtitle
    title_lines = _wrap_text(title, _TITLE_FONT, width * 0.8, draw)
    ty = height * 0.14
    for line in title_lines[:2]:
        draw.text((width / 2, ty), line, fill=_INK, font=_TITLE_FONT, anchor="mm")
        ty += 36
 
    draw.text((width / 2, height * 0.90), subtitle, fill=_TEAL, font=_SUBTITLE_FONT, anchor="mm")
 
    return np.array(img)
 
 
_LEADING_PHRASES = (
    "explain ", "what is ", "what are ", "how does ", "how do ",
    "why is ", "why does ", "tell me about ", "describe ",
)
 
 
def _extract_topic(query: str) -> str:
    """'Explain backpropagation' -> 'backpropagation', so the generated
    transcript reads as prose instead of echoing the raw question.
    """
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
    """Scene-by-scene transcript with estimated timestamps (~140 wpm speaking
    pace), rendered as clean HTML rather than raw text.
    """
    rows = []
    t = 0.0
    for i, scene in enumerate(scenes, start=1):
        words = len(scene["narration"].split())
        duration = max(6.0, words / 2.3)
        start, end = _fmt_timestamp(t), _fmt_timestamp(t + duration)
        rows.append(f"""
        <div class="em-transcript-scene">
          <div class="em-transcript-head">
            <span class="em-transcript-title">Scene {i} — {scene['title']}</span>
            <span class="em-transcript-time">{start}–{end}</span>
          </div>
          <p class="em-transcript-body">{scene['narration']}</p>
        </div>
        """)
        t += duration
    return f"<div class='em-transcript'>{''.join(rows)}</div>"
 
 
backend = MockBackend()
_retriever = Retriever() if RAG_AVAILABLE else None
 
 
def _resolve_kb_id(kb_state) -> str:
    return kb_state or DEFAULT_KB_ID
 
 
def _real_research_thought(job: "MockJob") -> str:
    """Called from the 'researching' progress step -- runs your actual
    hybrid BM25+embedding+rerank retrieval and surfaces a real result
    (or an honest 'nothing indexed' message) in the agent thoughts panel.
    """
    kb_id = _resolve_kb_id(job.kb_id)
    try:
        hits = _retriever.query(job.query, top_k=1, kb_id=kb_id)
    except Exception as e:
        return f"→ kb_search('{job.query[:40]}') failed: {e}"
 
    if not hits:
        return f"→ kb_search('{job.query[:40]}') → 0 results (no docs indexed in kb='{kb_id}')"
 
    top = hits[0]
    snippet = top["text"][:90].replace("\n", " ")
    return f"→ kb_search('{job.query[:40]}') → top hit ({top['score']:.2f}) {top['source']} p.{top['page']}: \"{snippet}…\""
 
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
 
 
def handle_send(message, history, kb_state, voice_state):
    if not message or not message.strip():
        yield history, "", "", gr.update(visible=False), gr.update(visible=False)
        return
 
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "_starting…_"},
    ]
    yield history, "**Starting…**", "", gr.update(visible=False), gr.update(visible=False)
 
    job = backend.create_job(message, kb_id=kb_state, voice=voice_state)
    thoughts = []
 
    for label, thought, frac in backend.stream_progress(job):
        pct = int(frac * 100)
        status_md = f"<div class='em-progress-label'>{label} · {pct}%</div>"
        if thought:
            thoughts.append(thought)
        history[-1]["content"] = f"_{label}…_"
        yield history, status_md, "\n".join(thoughts), gr.update(visible=False), gr.update(visible=False)
 
    scenes = backend.build_scenes(message)
    transcript_html = format_transcript(scenes)
 
    try:
        video_path = backend.render_preview_video(job)
        video_update = gr.update(value=video_path, visible=True)
        reply = "Here's your explainer video 🎬"
    except Exception as e:  # noqa: BLE001 - keep the app alive even if video synthesis fails
        video_update = gr.update(visible=False)
        reply = f"⚠️ Script generated, but the video preview failed to render: {e}"
 
    history[-1]["content"] = reply
    yield (
        history,
        "<div class='em-progress-label'>done · 100%</div>",
        "\n".join(thoughts),
        video_update,
        gr.update(value=transcript_html, visible=True),
    )
 
 
# ============================================================================
# KNOWLEDGE BASE TAB LOGIC
# ============================================================================
 
MAX_KB_SLOTS = 15  # fixed number of list rows; hidden/shown based on doc count
                    # (avoids gr.render, which isn't available in gradio==4.20.0)
 
 
def refresh_kb_slots(kb_state):
    """Build the full set of gr.update()s for every KB list slot + empty
    state, reading from the REAL registry on disk (tools/rag/registry.py),
    not a mock.
    """
    if not RAG_AVAILABLE:
        docs = []
    else:
        kb_id = _resolve_kb_id(kb_state)
        docs = list(reversed(list_documents(kb_id=kb_id)))
 
    updates = []
    for i in range(MAX_KB_SLOTS):
        if i < len(docs):
            d = docs[i]
            uploaded = d["uploaded_at"].replace("T", " ")[:16]
            label = f"<span class='em-kb-label'><strong>{d['filename']}</strong> — {d['chunks']} chunks · {uploaded}</span>"
            updates += [gr.update(visible=True), gr.update(value=label), d["doc_id"]]
        else:
            updates += [gr.update(visible=False), gr.update(value=""), ""]
    updates.append(gr.update(visible=(len(docs) == 0)))
    return updates
 
 
def handle_upload(files, kb_state):
    if not RAG_AVAILABLE:
        return refresh_kb_slots(kb_state) + [gr.update(value=f"⚠️ {_RAG_IMPORT_ERROR_MSG}")]
    if not files:
        return refresh_kb_slots(kb_state) + [gr.update()]
 
    kb_id = _resolve_kb_id(kb_state)
    ok, errors = 0, []
    for f in files:
        path = f.name if hasattr(f, "name") else str(f)
        try:
            result = ingest_file(path, kb_id=kb_id)
            ok += 1 if result["status"] in ("indexed", "duplicate") else 0
        except Exception as e:  # noqa: BLE001 - surface per-file failures, keep processing the rest
            errors.append(f"{Path(path).name}: {e}")
 
    _retriever.invalidate(kb_id)  # force next query to re-read the fresh index
 
    status = f"✓ Indexed {ok} file(s)"
    if errors:
        status += " · " + "; ".join(errors)
    return refresh_kb_slots(kb_state) + [gr.update(value=status)]
 
 
def handle_slot_delete(doc_id, kb_state):
    if RAG_AVAILABLE and doc_id:
        kb_id = _resolve_kb_id(kb_state)
        delete_document(doc_id, kb_id=kb_id)
        _retriever.invalidate(kb_id)
    return refresh_kb_slots(kb_state)
 
 
def handle_test_query(query_text, kb_state):
    if not query_text or not query_text.strip():
        return "Type a question above to preview retrieval."
    if not RAG_AVAILABLE:
        return f"⚠️ {_RAG_IMPORT_ERROR_MSG}"
 
    kb_id = _resolve_kb_id(kb_state)
    try:
        hits = _retriever.query(query_text, top_k=5, kb_id=kb_id)
    except Exception as e:  # noqa: BLE001 - show retrieval failures in-panel, don't crash the UI
        return f"⚠️ Retrieval failed: {e}"
 
    if not hits:
        return "_No results — upload a document above first, or try a different question._"
 
    lines = [f"**{h['score']:.2f}** · `{h['source']} p.{h['page']}`\n> {h['text']}" for h in hits]
    return "\n\n---\n\n".join(lines)
 
 
def handle_reindex(kb_state):
    if not RAG_AVAILABLE:
        return refresh_kb_slots(kb_state) + [gr.update(value=f"⚠️ {_RAG_IMPORT_ERROR_MSG}")]
    kb_id = _resolve_kb_id(kb_state)
    result = reindex_all(kb_id=kb_id)
    _retriever.invalidate(kb_id)
    return refresh_kb_slots(kb_state) + [gr.update(value=f"✓ Reindexed {result['chunk_count']} chunks")]
 
 
# ============================================================================
# BUILD APP
# ============================================================================
 
def build_app() -> gr.Blocks:
    with gr.Blocks(theme=THEME, css=CUSTOM_CSS, head=HEAD_HTML, title="EduManim") as demo:
 
        kb_state = gr.State(value=None)
        voice_state = gr.State(value="Narrator (neutral)")
 
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
 
                send_btn.click(
                    handle_send,
                    [msg_box, chatbot, kb_state, voice_state],
                    [chatbot, progress_md, thoughts_box, video_player, transcript_html],
                ).then(lambda: "", None, msg_box)
                msg_box.submit(
                    handle_send,
                    [msg_box, chatbot, kb_state, voice_state],
                    [chatbot, progress_md, thoughts_box, video_player, transcript_html],
                ).then(lambda: "", None, msg_box)
 
            # ========================= KNOWLEDGE BASE ========================
            with gr.Tab("📚 Knowledge Base"):
                gr.Markdown("Upload PDFs, Markdown, or text files. EduManim retrieves relevant passages before answering.")
                with gr.Row():
                    uploader = gr.File(label="Drop files here", file_count="multiple", scale=2)
                    with gr.Column(scale=1):
                        upload_status = gr.Markdown("")
                        reindex_btn = gr.Button("Reindex all", variant="secondary")
 
                gr.Markdown("**Indexed documents**")
 
                kb_empty_state = gr.HTML(
                    "<div class='em-empty'>No documents yet — upload a file above to get started.</div>",
                    visible=True,
                )
 
                kb_slot_rows, kb_slot_labels, kb_slot_ids, kb_slot_delete_btns = [], [], [], []
                for _ in range(MAX_KB_SLOTS):
                    with gr.Row(visible=False, elem_classes=["em-kb-row"]) as slot_row:
                        slot_label = gr.HTML("")
                        slot_id = gr.State("")
                        slot_delete = gr.Button("Delete", size="sm", variant="secondary", scale=0, min_width=90)
                    kb_slot_rows.append(slot_row)
                    kb_slot_labels.append(slot_label)
                    kb_slot_ids.append(slot_id)
                    kb_slot_delete_btns.append(slot_delete)
 
                kb_slot_outputs = []
                for row, label, sid in zip(kb_slot_rows, kb_slot_labels, kb_slot_ids):
                    kb_slot_outputs += [row, label, sid]
                kb_slot_outputs.append(kb_empty_state)
 
                gr.Markdown("**Test retrieval** — preview what the agent would see, before asking a full question.")
                with gr.Row():
                    test_query_box = gr.Textbox(placeholder="e.g. What is bonded labour?", show_label=False, scale=3)
                    test_query_btn = gr.Button("Preview", scale=1)
                test_query_out = gr.Markdown("")
 
                uploader.upload(handle_upload, [uploader, kb_state], kb_slot_outputs + [upload_status])
                for sid, btn in zip(kb_slot_ids, kb_slot_delete_btns):
                    btn.click(handle_slot_delete, [sid, kb_state], kb_slot_outputs)
                test_query_btn.click(handle_test_query, [test_query_box, kb_state], [test_query_out])
                reindex_btn.click(handle_reindex, [kb_state], kb_slot_outputs + [upload_status])
 
            # ============================ MY VIDEOS ==========================
            with gr.Tab("🎬 My Videos"):
                gr.HTML("<div class='em-empty'>No videos yet — generate one from the Chat tab and it'll show up here.</div>")
 
            # ============================ SETTINGS ===========================
            with gr.Tab("⚙️ Settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Voice**")
                        voice_radio = gr.Radio(
                            ["Narrator (neutral)", "Female", "Male"],
                            value="Narrator (neutral)",
                            show_label=False,
                        )
                        gr.Markdown("**Video quality**")
                        gr.Radio(["720p", "1080p"], value="720p", show_label=False)
                    with gr.Column():
                        gr.Markdown("**Agent verbosity**")
                        gr.Radio(["Quiet", "Normal", "Show agent thoughts"], value="Normal", show_label=False)
                        gr.Markdown("**Theme**")
                        gr.Radio(["Dark (default)", "Light"], value="Dark (default)", show_label=False)
 
                voice_radio.change(lambda v: v, [voice_radio], [voice_state])
 
        demo.load(refresh_kb_slots, [kb_state], kb_slot_outputs)
 
    return demo
 
 
if __name__ == "__main__":
    app = build_app()
    app.queue().launch()