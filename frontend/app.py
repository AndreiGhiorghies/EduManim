 
import inspect
import random
import time
import uuid
from datetime import datetime, timedelta
 
import gradio as gr
 
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
    """Standalone dev backend — no network, no GPU, just realistic timing."""
 
    def __init__(self):
        self.jobs: dict[str, MockJob] = {}
        self.kb_docs: dict[str, dict] = {}
        self.videos: list[dict] = []
 
    def create_job(self, query: str, kb_id: str | None = None, voice: str = "Narrator (neutral)") -> MockJob:
        job = MockJob(query, kb_id, voice)
        self.jobs[job.job_id] = job
        return job
 
    def stream_progress(self, job: MockJob):
        """Yields (status_line, thought_line, progress_fraction) tuples."""
        for status, label, frac in PROGRESS_STEPS:
            time.sleep(random.uniform(0.35, 0.7))
            thought = THOUGHT_LINES.get(status, "")
            if "{q}" in thought:
                thought = thought.format(q=job.query[:40])
            if "{n}" in thought:
                n = label.split(" ")[2] if "scene" in label else "1"
                thought = thought.format(n=n)
            if "{voice}" in thought:
                thought = thought.format(voice=job.voice)
            yield label, thought, frac
 
    def upload_doc(self, filename: str) -> dict:
        doc_id = str(uuid.uuid4())[:8]
        chunks = random.randint(8, 60)
        entry = {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": chunks,
            "uploaded_at": datetime.now().strftime("%b %d, %H:%M"),
        }
        self.kb_docs[doc_id] = entry
        return entry
 
    def delete_doc(self, doc_id: str) -> None:
        self.kb_docs.pop(doc_id, None)
 
    def docs_newest_first(self) -> list[dict]:
        return list(self.kb_docs.values())[::-1]
 
 
backend = MockBackend()
 
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
 
/* ---------- Video result card ---------- */
.em-video-card {
  border: 1px solid #1C2740; border-radius: 14px; overflow: hidden;
  background: linear-gradient(160deg, #101A30, #0B1120);
}
.em-video-thumb {
  height: 190px; display: flex; align-items: center; justify-content: center;
  background:
    radial-gradient(circle at 30% 30%, rgba(20,184,166,0.18), transparent 60%),
    radial-gradient(circle at 70% 70%, rgba(30,58,138,0.35), transparent 60%);
  position: relative;
}
.em-play-btn {
  width: 56px; height: 56px; border-radius: 50%;
  background: var(--em-teal); display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 24px rgba(20,184,166,0.5);
}
.em-video-meta { padding: 14px 18px; }
.em-video-meta .em-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: var(--em-ink); font-size: 15px; margin-bottom: 4px; }
.em-video-meta .em-sub { color: var(--em-muted); font-size: 12.5px; }
 
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
"""
 
# ============================================================================
# CHAT TAB LOGIC
# ============================================================================
 
 
def handle_send(message, history, kb_state, voice_state):
    if not message or not message.strip():
        yield history, "", "", gr.update(visible=False)
        return
 
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "_starting…_"},
    ]
    yield history, "**Starting…**", "", gr.update(visible=False)
 
    job = backend.create_job(message, kb_id=kb_state, voice=voice_state)
    thoughts = []
 
    for label, thought, frac in backend.stream_progress(job):
        pct = int(frac * 100)
        status_md = f"<div class='em-progress-label'>{label} · {pct}%</div>"
        if thought:
            thoughts.append(thought)
        history[-1]["content"] = f"_{label}…_"
        yield history, status_md, "\n".join(thoughts), gr.update(visible=False)
 
    video_card = f"""
    <div class="em-video-card">
      <div class="em-video-thumb">
        <div class="em-play-btn">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="#0B1120"><path d="M8 5v14l11-7z"/></svg>
        </div>
      </div>
      <div class="em-video-meta">
        <div class="em-title">{message[:60]}</div>
        <div class="em-sub">2:14 · 1080p · voiced by {job.voice} · rendered on our AMD Radeon GPU</div>
      </div>
    </div>
    """
    history[-1]["content"] = "Here's your explainer video 🎬"
    yield history, "<div class='em-progress-label'>done · 100%</div>", "\n".join(thoughts), gr.update(value=video_card, visible=True)
 
 
# ============================================================================
# KNOWLEDGE BASE TAB LOGIC
# ============================================================================
 
MAX_KB_SLOTS = 15  # fixed number of list rows; hidden/shown based on doc count
                    # (avoids gr.render, which isn't available in gradio==4.20.0)
 
 
def refresh_kb_slots():
    """Build the full set of gr.update()s for every KB list slot + empty state.
    Returns a flat list: [row_vis, label_md, doc_id, ...] * MAX_KB_SLOTS, then empty_state_vis.
    """
    docs = backend.docs_newest_first()
    updates = []
    for i in range(MAX_KB_SLOTS):
        if i < len(docs):
            d = docs[i]
            label = f"<span class='em-kb-label'><strong>{d['filename']}</strong> — {d['chunks']} chunks · {d['uploaded_at']}</span>"
            updates += [gr.update(visible=True), gr.update(value=label), d["doc_id"]]
        else:
            updates += [gr.update(visible=False), gr.update(value=""), ""]
    updates.append(gr.update(visible=(len(docs) == 0)))
    return updates
 
 
def handle_upload(files, kb_state):
    if not files:
        return refresh_kb_slots() + [gr.update()]
    for f in files:
        name = f.name.split("/")[-1] if hasattr(f, "name") else str(f)
        backend.upload_doc(name)
    status = f"✓ Indexed {len(files)} file(s)"
    return refresh_kb_slots() + [gr.update(value=status)]
 
 
def handle_slot_delete(doc_id):
    if doc_id:
        backend.delete_doc(doc_id)
    return refresh_kb_slots()
 
 
def handle_test_query(query_text):
    if not query_text or not query_text.strip():
        return "Type a question above to preview retrieval."
    fake_hits = [
        {"text": "…relevant passage would appear here, pulled from your uploaded docs…", "source": "example.pdf", "page": 3, "score": 0.91},
        {"text": "…a second supporting passage, ranked lower…", "source": "example.pdf", "page": 7, "score": 0.78},
    ]
    lines = [f"**{h['score']:.2f}** · `{h['source']} p.{h['page']}`\n> {h['text']}" for h in fake_hits]
    return "\n\n".join(lines)
 
 
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
            planned, researched, rendered, and voiced entirely on our GPU.</p>
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
                        video_html = gr.HTML(visible=False)
                        with gr.Accordion("Agent thoughts", open=False):
                            thoughts_box = gr.Textbox(
                                value="",
                                lines=10,
                                show_label=False,
                                interactive=False,
                                elem_classes=["em-terminal"],
                            )
 
                send_btn.click(
                    handle_send, [msg_box, chatbot, kb_state, voice_state], [chatbot, progress_md, thoughts_box, video_html]
                ).then(lambda: "", None, msg_box)
                msg_box.submit(
                    handle_send, [msg_box, chatbot, kb_state, voice_state], [chatbot, progress_md, thoughts_box, video_html]
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
                    btn.click(handle_slot_delete, [sid], kb_slot_outputs)
                test_query_btn.click(handle_test_query, [test_query_box], [test_query_out])
 
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
 
    return demo
 
 
if __name__ == "__main__":
    app = build_app()
    app.queue().launch()