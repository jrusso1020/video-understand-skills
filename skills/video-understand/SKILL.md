---
name: video-understand
description: |
  Video understanding and transcription with intelligent multi-provider fallback. Use when: (1) Transcribing video or audio content, (2) Understanding video content including visual elements and scenes, (3) Analyzing YouTube videos by URL, (4) Extracting information from local video files, (5) Getting timestamps, summaries, or answering questions about video content. Automatically selects the best available provider based on configured API keys - prefers full video understanding (Gemini/OpenRouter) over ASR-only providers.
---

# Video Understanding

Multi-provider video understanding with automatic fallback based on available API keys.

## Quick Start

1. Check available providers:
```bash
python scripts/check_providers.py
```

2. Process a video:
```bash
# YouTube URL (Gemini can process directly)
python scripts/process_video.py "https://youtube.com/watch?v=..."

# Local file
python scripts/process_video.py /path/to/video.mp4

# With specific provider
python scripts/process_video.py --provider openai video.mp4
```

## Provider Hierarchy

The skill automatically selects the best available provider:

| Priority | Provider | Capability | Env Var |
|----------|----------|------------|---------|
| 1 | Gemini | Full video (visual + audio) | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| 2 | Vertex AI | Full video | `GOOGLE_APPLICATION_CREDENTIALS` |
| 3 | OpenRouter | Full video (via Gemini) | `OPENROUTER_API_KEY` |
| 4 | OpenAI Whisper | ASR only | `OPENAI_API_KEY` |
| 5 | AssemblyAI | ASR + speaker labels | `ASSEMBLYAI_API_KEY` |
| 6 | Deepgram | ASR | `DEEPGRAM_API_KEY` |
| 7 | Groq Whisper | ASR (fast) | `GROQ_API_KEY` |
| 8 | Local Whisper | ASR (offline) | None (requires `whisper` CLI) |

**Full video providers** can analyze visual content, read on-screen text, describe scenes, and transcribe audio.
**ASR providers** transcribe audio only - video must be converted to audio first.

## Workflow

```
Input (YouTube URL / Local File / URL)
    │
    ├─► Check providers (check_providers.py)
    │
    ├─► Full Video Provider Available?
    │   ├─► YES: Send directly to Gemini/OpenRouter
    │   │        └─► Return: transcript + visual analysis + timestamps
    │   │
    │   └─► NO: ASR Fallback
    │            ├─► Download video if URL (yt-dlp)
    │            ├─► Extract audio (ffmpeg)
    │            └─► Transcribe with best ASR provider
    │                 └─► Return: transcript + timestamps
    │
    └─► Structured JSON Output
```

## Output Format

All providers return consistent JSON. See [output-format.md](references/output-format.md) for full schema.

```json
{
  "provider": "gemini",
  "capability": "full_video",
  "source": {"type": "youtube", "url": "..."},
  "transcript": [
    {"start": 0.0, "end": 2.5, "text": "Hello and welcome"}
  ],
  "visual_analysis": {
    "scenes": [...],
    "on_screen_text": [...],
    "key_moments": [...]
  },
  "summary": "...",
  "metadata": {"duration_seconds": 120, "language": "en"}
}
```

## Quick Reference

| Task | Reference |
|------|-----------|
| Use Gemini for video | [gemini.md](references/gemini.md) |
| Use OpenRouter | [openrouter.md](references/openrouter.md) |
| ASR providers (Whisper, AssemblyAI, etc.) | [asr-providers.md](references/asr-providers.md) |
| Output JSON schema | [output-format.md](references/output-format.md) |
| Video sources & downloading | [video-sources.md](references/video-sources.md) |

## Common Tasks

### Transcribe YouTube Video
```python
# Gemini handles YouTube URLs natively
from scripts.gemini_video import process_youtube
result = process_youtube("https://youtube.com/watch?v=...", prompt="Transcribe this video")
```

### Analyze Video Content
```python
# Ask questions about video content
result = process_youtube(url, prompt="What products are shown? List with timestamps.")
```

### ASR-Only Transcription
```python
# Force ASR provider when you only need transcript
from scripts.process_video import process_video
result = process_video("video.mp4", provider="openai", asr_only=True)
```

## Requirements

**For full video understanding:**
- Gemini: `pip install google-generativeai`
- OpenRouter: `pip install openai` (uses OpenAI-compatible API)

**For ASR fallback:**
- `yt-dlp` for downloading videos
- `ffmpeg` for audio extraction
- Provider SDK (`openai`, `assemblyai`, `deepgram-sdk`, etc.)

**For local Whisper:**
- `pip install openai-whisper` or Whisper.cpp
