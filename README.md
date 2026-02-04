# Video Understand Skills

Claude Code skills for video understanding and transcription with intelligent multi-provider fallback.

## Features

- **Full Video Understanding** (visual + audio) via Gemini or OpenRouter
- **ASR Transcription** via OpenAI Whisper, AssemblyAI, Deepgram, Groq, or local Whisper
- **Automatic Provider Selection** based on available API keys
- **Multiple Input Sources**: YouTube URLs, local files, and video URLs

## Provider Hierarchy

| Priority | Provider | Capability | Env Variable |
|----------|----------|------------|--------------|
| 1 | Gemini | Full video (visual + audio) | `GEMINI_API_KEY` |
| 2 | Vertex AI | Full video | `GOOGLE_APPLICATION_CREDENTIALS` |
| 3 | OpenRouter | Full video (via Gemini) | `OPENROUTER_API_KEY` |
| 4 | OpenAI Whisper | ASR only | `OPENAI_API_KEY` |
| 5 | AssemblyAI | ASR + analysis | `ASSEMBLYAI_API_KEY` |
| 6 | Deepgram | ASR | `DEEPGRAM_API_KEY` |
| 7 | Groq Whisper | ASR (fast) | `GROQ_API_KEY` |
| 8 | Local Whisper | ASR (offline) | None |

## Installation

### Using skills CLI (recommended)

```bash
npx skills add your-username/video-understand-skills -a claude-code -g
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/your-username/video-understand-skills.git

# Symlink to Claude Code skills (global)
ln -s $(pwd)/video-understand-skills/skills/video-understand ~/.claude/skills/video-understand

# Or project-specific
ln -s $(pwd)/video-understand-skills/skills/video-understand .claude/skills/video-understand
```

## Requirements

### For full video understanding (Gemini/OpenRouter)

```bash
pip install google-generativeai  # For Gemini
pip install openai               # For OpenRouter
```

### For ASR fallback

```bash
# Video downloading
pip install yt-dlp
# or
brew install yt-dlp

# Audio processing (usually pre-installed)
brew install ffmpeg

# Provider SDKs (install as needed)
pip install openai           # OpenAI Whisper
pip install assemblyai       # AssemblyAI
pip install deepgram-sdk     # Deepgram
pip install groq             # Groq
pip install openai-whisper   # Local Whisper
```

## Usage

### Check Available Providers

```bash
python3 skills/video-understand/scripts/check_providers.py
```

### Process a Video

```bash
# YouTube URL
python3 skills/video-understand/scripts/process_video.py "https://youtube.com/watch?v=..."

# Local file
python3 skills/video-understand/scripts/process_video.py video.mp4

# Force specific provider
python3 skills/video-understand/scripts/process_video.py --provider openai video.mp4

# ASR-only mode
python3 skills/video-understand/scripts/process_video.py --asr-only video.mp4
```

## Output Format

All providers return consistent JSON:

```json
{
  "provider": "gemini",
  "capability": "full_video",
  "source": {"type": "youtube", "url": "..."},
  "transcript": [
    {"start": 0.0, "end": 2.5, "text": "Hello"}
  ],
  "text": "Full transcript...",
  "visual_analysis": {
    "scenes": [...],
    "on_screen_text": [...]
  },
  "summary": "..."
}
```

## License

MIT
