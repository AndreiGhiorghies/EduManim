# Track B — Video & Audio Pipeline

Three standalone CLI tools that together turn a Manim scene script + narration text into a
final assembled video: **Manim** (render) → **TTS** (narrate) → **FFmpeg** (assemble).

Each tool is independent and can be tested/run without the other two.

---

## 1. Manim Tool (`tools/manim/`)

Validates and renders Manim scene code (LLM-generated or hand-written) into MP4 clips.

### `validate` — check if generated code is safe/correct before rendering
```bash
python -m tools.manim.cli validate <scene_file.py>
```
- Exit code `0` = valid, `1` = invalid
- Checks: AST parse, import allowlist, forbidden patterns (`os.system`, `eval`, `exec`, `subprocess`, network calls), Scene class detection, naming pattern

### `render` — render a validated scene file to MP4
```bash
python -m tools.manim.cli render <scene_file.py> <SceneClassName> --output <path.mp4>
```
Returns JSON:
```json
{"success": true, "video_path": "/tmp/test_scene.mp4", "duration_sec": 2.0, "render_time_sec": 10.07, "error": null}
```
- Timeout: 90s
- On failure, `success: false` and `error` contains the reason

### `render-template` — generate + render from a pre-built template
```bash
python -m tools.manim.cli render-template <template_name> --params '{"title":"...", "formula":"..."}' --output <path.mp4>
```
Available templates: `definition`, `formula`, `graph`, `comparison`, `code`

### Fallback
If LLM-generated code repeatedly fails validation/render, use `tools/manim/fallback.py` to
generate a static text-scene from a `visual_hint` string — guarantees Track A always gets *some* video output.

---

## 2. TTS Tool (`tools/tts/`)

Converts narration text (including LaTeX/math) into speech using Coqui XTTS v2 with voice cloning.

### `synthesize` — text → audio file
```bash
python -m tools.tts.cli synthesize "<text>" --output <path.wav> --voice <voice_name>
```
Returns JSON:
```json
{"audio_path": "/tmp/test_synth.wav", "duration_sec": 4.77}
```
- `--voice` options: `female`, `male`, `narrator` (default), or a path to a custom reference `.wav`
- Text is auto-cleaned (LaTeX → spoken form, e.g. `\sum` → "sum") before synthesis
- Long text is auto-chunked (>220 chars) and concatenated
- On OOM, automatically splits and retries; on total failure, falls back to a silent placeholder track (same duration) so the pipeline never hard-crashes

### `list-voices` — show available voices
```bash
python -m tools.tts.cli list-voices
```

### `preview` — quick sample of a voice
```bash
python -m tools.tts.cli preview <voice_name> --output <path.wav>
```

**Voice samples location:** `tools/tts/samples/voice_{female,male,narrator}.wav` — must be real speech (10s+), not silence. Raw source recordings kept alongside as `*_raw.mp3` for reference.

**Note:** Runs on CPU or GPU (auto-detects `torch.cuda.is_available()`). CPU synthesis is ~8x real-time (slow) — for full-length narrations, run on the Radeon Cloud GPU.

---

## 3. FFmpeg Tool (`tools/ffmpeg/`)

Assembles rendered scene clips + narration audio into the final video.

### `assemble` — mux + concat scenes with audio
```bash
python -m tools.ffmpeg.cli assemble \
  --scenes scene1.mp4 scene2.mp4 ... \
  --audios audio1.wav audio2.wav ... \
  --output final.mp4
```
Returns JSON:
```json
{"video_path": "/tmp/final_test.mp4", "duration_sec": 4.77, "size_mb": 0.12}
```
- Number of `--scenes` must match number of `--audios` (paired by index)
- Each scene's original audio track is replaced with its matching narration
- **Duration mismatch handling:**
  - Narration shorter than scene → padded with silence
  - Narration longer than scene → scene extended via `tpad` filter
- All scenes are concatenated in order using the concat demuxer

### `info` — inspect a video file
```bash
python -m tools.ffmpeg.cli info <video_path>
```
Returns JSON: `{"duration": ..., "width": ..., "height": ..., "fps": ..., "codec": "h264"}`

---

## Full Pipeline Example (how Track A should call this)

```bash
# 1. Render each scene
python -m tools.manim.cli render scene1.py Scene1 --output /tmp/s1.mp4

# 2. Synthesize narration for each scene
python -m tools.tts.cli synthesize "Explanation text..." --output /tmp/a1.wav --voice narrator

# 3. Assemble everything into the final video
python -m tools.ffmpeg.cli assemble --scenes /tmp/s1.mp4 --audios /tmp/a1.wav --output /tmp/final.mp4
```

All three CLIs communicate via **JSON on stdout** and a **process exit code** — easy to wrap
in Python subprocess calls from the agent (Track A) with no shared state or imports required.

## Known Quirks
- Manim: LaTeX/MathTex rendering requires TeX Live installed (`texlive-latex-extra`, etc.)
- TTS: first run downloads the XTTS v2 model (~2GB); subsequent runs use the cached copy
- FFmpeg: `assemble` requires `ffmpeg` on PATH