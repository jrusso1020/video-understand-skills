# Video Understand Skills

Claude Code skills for video understanding and transcription with intelligent multi-provider fallback.

## Features

- **Full Video Understanding** (visual + audio) via Gemini or OpenRouter
- **ASR Transcription** via OpenAI Whisper, AssemblyAI, Deepgram, Groq, or local Whisper
- **Automatic Provider Selection** based on available API keys
- **Model Selection** per provider with sensible defaults
- **Robust Path Handling** for macOS special characters and unicode filenames
- **Multiple Input Sources**: YouTube URLs, local files, and video URLs

## Provider Hierarchy

| Priority | Provider | Capability | Env Variable | Default Model |
|----------|----------|------------|--------------|---------------|
| 1 | Gemini | Full video | `GEMINI_API_KEY` | gemini-2.0-flash |
| 2 | Vertex AI | Full video | `GOOGLE_APPLICATION_CREDENTIALS` | gemini-2.0-flash |
| 3 | OpenRouter | Full video | `OPENROUTER_API_KEY` | google/gemini-3-flash-preview |
| 4 | OpenAI Whisper | ASR only | `OPENAI_API_KEY` | whisper-1 |
| 5 | AssemblyAI | ASR + analysis | `ASSEMBLYAI_API_KEY` | best |
| 6 | Deepgram | ASR | `DEEPGRAM_API_KEY` | nova-2 |
| 7 | Groq Whisper | ASR (fast) | `GROQ_API_KEY` | whisper-large-v3-turbo |
| 8 | Local Whisper | ASR (offline) | None | base |

## Installation

### Using skills CLI (recommended)

```bash
npx skills add jrusso1020/video-understand-skills -a claude-code -g
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/jrusso1020/video-understand-skills.git

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
# Video downloading and processing
brew install yt-dlp ffmpeg

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

# Custom prompt
python3 skills/video-understand/scripts/process_video.py video.mp4 -p "List all products shown"

# Force specific provider and model
python3 skills/video-understand/scripts/process_video.py video.mp4 --provider openrouter -m google/gemini-3-pro-preview

# ASR-only mode (skip visual analysis)
python3 skills/video-understand/scripts/process_video.py video.mp4 --asr-only

# Quiet mode (no progress output)
python3 skills/video-understand/scripts/process_video.py video.mp4 -q

# Save to file
python3 skills/video-understand/scripts/process_video.py video.mp4 -o result.json
```

### List Available Models

```bash
python3 skills/video-understand/scripts/process_video.py --list-models
```

## Output Format

All providers return consistent JSON:

```json
{
  "source": {
    "type": "youtube",
    "path": "https://youtube.com/...",
    "duration_seconds": 120.5,
    "size_mb": 15.2
  },
  "provider": "openrouter",
  "model": "google/gemini-3-flash-preview",
  "capability": "full_video",
  "response": "The video shows...",
  "transcript": [
    {"start": 0.0, "end": 2.5, "text": "Hello and welcome"}
  ],
  "text": "Full transcript as single string..."
}
```

## CLI Options

```
python3 process_video.py [OPTIONS] SOURCE

Arguments:
  SOURCE              YouTube URL, video URL, or local file path

Options:
  -p, --prompt TEXT   Custom prompt for video understanding
  --provider NAME     Force specific provider
  -m, --model NAME    Force specific model (use --list-models to see options)
  --asr-only          Force ASR-only mode (no visual analysis)
  -o, --output FILE   Output JSON file (default: stdout)
  -q, --quiet         Suppress progress messages
  --list-models       List available models and exit
  --list-providers    List available providers as JSON and exit
```

## License

MIT
