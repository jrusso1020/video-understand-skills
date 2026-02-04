#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main video processing script with automatic provider selection.
Handles YouTube URLs, local files, and other video URLs.

Features:
- Multi-provider support with automatic fallback
- Model selection per provider
- Robust path handling (macOS special characters, unicode, etc.)
- Progress output and verbose mode
- File size warnings for API limits
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

# Import provider check
from check_providers import get_available_providers

# === Default Models ===
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",      # Current production model
    "vertex": "gemini-2.5-flash",      # Current production model
    "openrouter": "google/gemini-3-flash-preview",  # Latest via OpenRouter
    "openai": "whisper-1",
    "groq": "whisper-large-v3-turbo",
    "assemblyai": "best",  # AssemblyAI auto-selects
    "deepgram": "nova-2",
    "local": "base",
}

# === Available Models ===
AVAILABLE_MODELS = {
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-pro"],
    "vertex": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-pro"],
    "openrouter": [
        "google/gemini-3-flash-preview",
        "google/gemini-3-pro-preview",
    ],
    "openai": ["whisper-1"],
    "groq": ["whisper-large-v3", "whisper-large-v3-turbo"],
    "assemblyai": ["best", "nano"],
    "deepgram": ["nova-2", "nova", "enhanced", "base"],
    "local": ["tiny", "base", "small", "medium", "large", "large-v3"],
}

# File size limits (in MB)
FILE_SIZE_LIMITS = {
    "openrouter": 50,  # Base64 encoding limit
    "openai": 25,      # Whisper API limit
    "groq": 25,        # Whisper API limit
}


def log(msg: str, verbose: bool = True):
    """Print status message to stderr."""
    if verbose:
        print(f"[video-understand] {msg}", file=sys.stderr)


def normalize_path(path: str) -> str:
    """
    Normalize a file path to handle macOS special characters, unicode, etc.

    Handles:
    - Unicode normalization (NFD -> NFC for macOS)
    - Tilde expansion (~)
    - Symlink resolution
    - Relative to absolute path conversion
    """
    if not path:
        return path

    # Expand user home directory
    path = os.path.expanduser(path)

    # Normalize unicode (macOS uses NFD, most systems expect NFC)
    path = unicodedata.normalize("NFC", path)

    # Convert to absolute path
    path = os.path.abspath(path)

    # Try to resolve the path (follows symlinks, normalizes case on case-insensitive FS)
    try:
        resolved = Path(path).resolve()
        if resolved.exists():
            return str(resolved)
    except (OSError, ValueError):
        pass

    return path


def normalize_whitespace(s: str) -> str:
    """Normalize all whitespace variants to regular spaces."""
    import re
    # Replace various Unicode whitespace characters with regular space
    # This handles: \u202f (narrow no-break space), \u00a0 (no-break space), etc.
    return re.sub(r'[\s\u00a0\u202f\u2007\u2008\u2009\u200a\u200b]+', ' ', s)


def find_file_fuzzy(path: str) -> str:
    """
    Try to find a file even if the exact path doesn't match.
    Handles macOS filename encoding issues including:
    - Unicode normalization (NFD vs NFC)
    - Special whitespace characters (narrow no-break space, etc.)
    - Case-insensitive matching
    """
    # First try the normalized path
    normalized = normalize_path(path)
    if os.path.isfile(normalized):
        return normalized

    # Try the original path
    if os.path.isfile(path):
        return path

    # If the file doesn't exist, try to find it in the parent directory
    parent = os.path.dirname(normalized) or "."
    basename = os.path.basename(normalized)

    if not os.path.isdir(parent):
        return normalized  # Can't search, return as-is

    # Normalize for comparison
    basename_nfc = unicodedata.normalize("NFC", basename)
    basename_nfd = unicodedata.normalize("NFD", basename)
    basename_ws = normalize_whitespace(basename_nfc)

    for entry in os.listdir(parent):
        entry_nfc = unicodedata.normalize("NFC", entry)
        entry_nfd = unicodedata.normalize("NFD", entry)
        entry_ws = normalize_whitespace(entry_nfc)

        # Exact match with unicode normalization
        if entry_nfc == basename_nfc or entry_nfd == basename_nfd:
            return os.path.join(parent, entry)

        # Match with whitespace normalization (handles \u202f, \u00a0, etc.)
        if entry_ws == basename_ws:
            return os.path.join(parent, entry)

        # Case-insensitive match
        if entry_nfc.lower() == basename_nfc.lower():
            return os.path.join(parent, entry)

        # Case-insensitive with whitespace normalization
        if entry_ws.lower() == basename_ws.lower():
            return os.path.join(parent, entry)

    return normalized


def is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube video."""
    patterns = [
        r'(youtube\.com/watch\?v=)',
        r'(youtu\.be/)',
        r'(youtube\.com/embed/)',
        r'(youtube\.com/v/)',
        r'(youtube\.com/shorts/)',
    ]
    return any(re.search(p, url) for p in patterns)


def get_file_size_mb(path: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


def download_video(url: str, output_dir: str, max_size_mb: int = None, verbose: bool = True) -> str:
    """Download video using yt-dlp."""
    log(f"Downloading video from: {url}", verbose)

    output_path = os.path.join(output_dir, "video.%(ext)s")

    # Build format string based on size limit
    if max_size_mb:
        format_str = f"best[ext=mp4][filesize<{max_size_mb}M]/best[ext=mp4][filesize<{max_size_mb * 2}M]/best"
    else:
        format_str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"

    cmd = [
        "yt-dlp",
        "-f", format_str,
        "-o", output_path,
        "--no-playlist",
        url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    # Find the downloaded file
    for f in os.listdir(output_dir):
        if f.startswith("video."):
            full_path = os.path.join(output_dir, f)
            size_mb = get_file_size_mb(full_path)
            log(f"Downloaded: {f} ({size_mb:.1f} MB)", verbose)
            return full_path

    raise FileNotFoundError("Downloaded video not found")


def convert_to_mp4(input_path: str, output_dir: str, verbose: bool = True) -> str:
    """Convert video to MP4 format if needed."""
    ext = Path(input_path).suffix.lower()

    # These formats are generally well-supported
    supported = {".mp4", ".mov", ".webm", ".m4v"}

    if ext in supported:
        return input_path

    log(f"Converting {ext} to MP4...", verbose)
    output_path = os.path.join(output_dir, "converted.mp4")

    cmd = [
        "ffmpeg", "-i", input_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"Conversion failed, using original: {result.stderr[:200]}", verbose)
        return input_path

    return output_path


def extract_audio(video_path: str, output_dir: str, verbose: bool = True) -> str:
    """Extract audio from video using ffmpeg."""
    log("Extracting audio...", verbose)

    audio_path = os.path.join(output_dir, "audio.mp3")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",           # No video
        "-acodec", "libmp3lame",
        "-ab", "128k",
        "-ar", "16000",  # 16kHz for Whisper
        "-y",            # Overwrite
        audio_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr}")

    return audio_path


def get_video_info(path: str) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size:stream=width,height",
        "-of", "json",
        path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}

    try:
        data = json.loads(result.stdout)
        return {
            "duration_seconds": float(data.get("format", {}).get("duration", 0)),
            "size_bytes": int(data.get("format", {}).get("size", 0)),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        return {}


# === Provider-specific processors ===

def process_with_gemini(source: str, prompt: str, model: str = None, is_url: bool = False, verbose: bool = True) -> dict:
    """Process video with Google Gemini."""
    import google.generativeai as genai

    model_name = model or DEFAULT_MODELS["gemini"]
    log(f"Processing with Gemini ({model_name})...", verbose)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)

    genai_model = genai.GenerativeModel(model_name)

    if is_url and is_youtube_url(source):
        # Gemini can handle YouTube URLs directly
        log("Sending YouTube URL directly to Gemini...", verbose)
        response = genai_model.generate_content([
            prompt,
            {"video_url": source}
        ])
    else:
        # Upload local file
        log("Uploading video to Gemini...", verbose)
        video_file = genai.upload_file(source)

        # Wait for processing
        import time
        while video_file.state.name == "PROCESSING":
            log("Waiting for Gemini to process video...", verbose)
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise RuntimeError(f"Video processing failed: {video_file.state.name}")

        log("Generating response...", verbose)
        response = genai_model.generate_content([prompt, video_file])

    return {
        "provider": "gemini",
        "model": model_name,
        "capability": "full_video",
        "response": response.text,
    }


def process_with_openrouter(video_path: str, prompt: str, model: str = None, verbose: bool = True) -> dict:
    """Process video with OpenRouter (using Gemini models)."""
    import base64
    from openai import OpenAI

    model_name = model or DEFAULT_MODELS["openrouter"]
    log(f"Processing with OpenRouter ({model_name})...", verbose)

    # Check file size
    size_mb = get_file_size_mb(video_path)
    limit = FILE_SIZE_LIMITS.get("openrouter", 50)
    if size_mb > limit:
        log(f"WARNING: File size ({size_mb:.1f} MB) exceeds recommended limit ({limit} MB)", verbose)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"]
    )

    # Read and encode video
    log(f"Encoding video ({size_mb:.1f} MB)...", verbose)
    with open(video_path, "rb") as f:
        video_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    # Get mime type
    ext = Path(video_path).suffix.lower()
    mime_types = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime", ".m4v": "video/x-m4v"}
    mime_type = mime_types.get(ext, "video/mp4")

    log("Sending to OpenRouter...", verbose)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{video_b64}"
                    }
                }
            ]
        }]
    )

    return {
        "provider": "openrouter",
        "model": model_name,
        "capability": "full_video",
        "response": response.choices[0].message.content,
    }


def process_with_openai_whisper(audio_path: str, model: str = None, verbose: bool = True) -> dict:
    """Transcribe audio with OpenAI Whisper API."""
    from openai import OpenAI

    model_name = model or DEFAULT_MODELS["openai"]
    log(f"Transcribing with OpenAI Whisper ({model_name})...", verbose)

    client = OpenAI()

    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model=model_name,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    segments = []
    if hasattr(transcript, 'segments') and transcript.segments:
        for seg in transcript.segments:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            })

    return {
        "provider": "openai",
        "model": model_name,
        "capability": "asr_only",
        "transcript": segments,
        "text": transcript.text,
        "language": getattr(transcript, 'language', None),
    }


def process_with_groq_whisper(audio_path: str, model: str = None, verbose: bool = True) -> dict:
    """Transcribe audio with Groq Whisper."""
    from groq import Groq

    model_name = model or DEFAULT_MODELS["groq"]
    log(f"Transcribing with Groq Whisper ({model_name})...", verbose)

    client = Groq()

    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model=model_name,
            file=f,
            response_format="verbose_json",
        )

    segments = []
    if hasattr(transcript, 'segments') and transcript.segments:
        for seg in transcript.segments:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            })

    return {
        "provider": "groq",
        "model": model_name,
        "capability": "asr_only",
        "transcript": segments,
        "text": transcript.text,
        "language": getattr(transcript, 'language', None),
    }


def process_with_assemblyai(audio_path: str, model: str = None, verbose: bool = True) -> dict:
    """Transcribe audio with AssemblyAI."""
    import assemblyai as aai

    model_name = model or DEFAULT_MODELS["assemblyai"]
    log(f"Transcribing with AssemblyAI ({model_name})...", verbose)

    aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]

    config = aai.TranscriptionConfig(
        speaker_labels=True,
        auto_chapters=True,
        speech_model=aai.SpeechModel.best if model_name == "best" else aai.SpeechModel.nano,
    )

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    segments = []
    if transcript.utterances:
        for utt in transcript.utterances:
            segments.append({
                "start": utt.start / 1000,
                "end": utt.end / 1000,
                "text": utt.text,
                "speaker": utt.speaker,
            })

    chapters = []
    if transcript.chapters:
        for ch in transcript.chapters:
            chapters.append({
                "start": ch.start / 1000,
                "end": ch.end / 1000,
                "headline": ch.headline,
                "summary": ch.summary,
            })

    return {
        "provider": "assemblyai",
        "model": model_name,
        "capability": "asr_only",
        "transcript": segments,
        "text": transcript.text,
        "chapters": chapters,
    }


def process_with_deepgram(audio_path: str, model: str = None, verbose: bool = True) -> dict:
    """Transcribe audio with Deepgram."""
    from deepgram import DeepgramClient, PrerecordedOptions

    model_name = model or DEFAULT_MODELS["deepgram"]
    log(f"Transcribing with Deepgram ({model_name})...", verbose)

    client = DeepgramClient(os.environ["DEEPGRAM_API_KEY"])

    with open(audio_path, "rb") as f:
        buffer_data = f.read()

    payload = {"buffer": buffer_data}

    options = PrerecordedOptions(
        model=model_name,
        smart_format=True,
        utterances=True,
        punctuate=True,
        diarize=True,
    )

    response = client.listen.prerecorded.v("1").transcribe_file(payload, options)
    result = response.to_dict()

    segments = []
    if "utterances" in result.get("results", {}):
        for utt in result["results"]["utterances"]:
            segments.append({
                "start": utt["start"],
                "end": utt["end"],
                "text": utt["transcript"],
                "speaker": utt.get("speaker"),
            })

    return {
        "provider": "deepgram",
        "model": model_name,
        "capability": "asr_only",
        "transcript": segments,
        "text": result["results"]["channels"][0]["alternatives"][0]["transcript"],
    }


def process_with_local_whisper(audio_path: str, model: str = None, verbose: bool = True) -> dict:
    """Transcribe audio with local Whisper."""
    model_name = model or DEFAULT_MODELS["local"]
    log(f"Transcribing with local Whisper ({model_name})...", verbose)

    try:
        import whisper

        log("Loading Whisper model...", verbose)
        model_obj = whisper.load_model(model_name)

        log("Transcribing...", verbose)
        result = model_obj.transcribe(audio_path)

        segments = []
        for seg in result["segments"]:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            })

        return {
            "provider": "local",
            "model": model_name,
            "capability": "asr_only",
            "transcript": segments,
            "text": result["text"],
            "language": result.get("language"),
        }
    except ImportError:
        # Fall back to CLI
        log("Using Whisper CLI...", verbose)
        cmd = ["whisper", audio_path, "--model", model_name, "--output_format", "json"]
        subprocess.run(cmd, check=True, capture_output=True)

        json_path = Path(audio_path).with_suffix(".json")
        with open(json_path) as f:
            result = json.load(f)

        segments = []
        for seg in result["segments"]:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            })

        return {
            "provider": "local",
            "model": model_name,
            "capability": "asr_only",
            "transcript": segments,
            "text": result["text"],
        }


def process_video(
    source: str,
    prompt: str = None,
    provider: str = None,
    model: str = None,
    asr_only: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Process a video with automatic provider selection.

    Args:
        source: YouTube URL, local file path, or video URL
        prompt: Prompt for video understanding (used by Gemini/OpenRouter)
        provider: Force specific provider (optional)
        model: Force specific model (optional)
        asr_only: Force ASR-only mode even if video understanding available
        verbose: Print progress messages

    Returns:
        dict with transcript, analysis, and metadata
    """

    # Default prompt
    if prompt is None:
        prompt = "Provide a detailed transcript with timestamps. Also describe any important visual elements, on-screen text, and key moments."

    # Check available providers
    providers = get_available_providers()

    if provider:
        if provider not in providers["all_providers"]:
            available = providers["all_providers"] or ["none"]
            raise ValueError(f"Provider '{provider}' not available. Available: {', '.join(available)}")
    elif asr_only:
        provider = providers["asr_only"][0] if providers["asr_only"] else None
    else:
        provider = providers["recommended"]

    if not provider:
        raise RuntimeError(
            "No video understanding providers available. "
            "Set one of: GEMINI_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY, etc."
        )

    log(f"Using provider: {provider}", verbose)

    # Determine source type
    is_url = source.startswith(("http://", "https://"))
    is_youtube = is_url and is_youtube_url(source)

    # Handle local files with path normalization
    if not is_url:
        source = find_file_fuzzy(source)
        is_local = os.path.isfile(source)
        if not is_local:
            raise FileNotFoundError(f"Source not found: {source}")
    else:
        is_local = False

    result = {
        "source": {
            "type": "youtube" if is_youtube else ("url" if is_url else "local"),
            "path": source,
        }
    }

    # Add video info for local files
    if is_local:
        info = get_video_info(source)
        if info:
            result["source"]["duration_seconds"] = info.get("duration_seconds")
            result["source"]["size_mb"] = round(info.get("size_bytes", 0) / (1024 * 1024), 2)

    # === Full Video Understanding ===
    if provider in providers["video_understanding"] and not asr_only:

        if provider == "gemini":
            if is_youtube:
                result.update(process_with_gemini(source, prompt, model=model, is_url=True, verbose=verbose))
            elif is_local:
                result.update(process_with_gemini(source, prompt, model=model, is_url=False, verbose=verbose))
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = download_video(source, tmpdir, verbose=verbose)
                    result.update(process_with_gemini(video_path, prompt, model=model, is_url=False, verbose=verbose))

        elif provider == "openrouter":
            if is_local:
                # Check if we need to convert format
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = convert_to_mp4(source, tmpdir, verbose=verbose)
                    result.update(process_with_openrouter(video_path, prompt, model=model, verbose=verbose))
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    max_size = FILE_SIZE_LIMITS.get("openrouter")
                    video_path = download_video(source, tmpdir, max_size_mb=max_size, verbose=verbose)
                    video_path = convert_to_mp4(video_path, tmpdir, verbose=verbose)
                    result.update(process_with_openrouter(video_path, prompt, model=model, verbose=verbose))

        elif provider == "vertex":
            raise NotImplementedError("Vertex AI support coming soon. Use GEMINI_API_KEY instead.")

    # === ASR-Only Providers ===
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            if is_url:
                video_path = download_video(source, tmpdir, verbose=verbose)
            else:
                video_path = source

            audio_path = extract_audio(video_path, tmpdir, verbose=verbose)

            # Check file size for providers with limits
            if provider in FILE_SIZE_LIMITS:
                size_mb = get_file_size_mb(audio_path)
                limit = FILE_SIZE_LIMITS[provider]
                if size_mb > limit:
                    log(f"WARNING: Audio file ({size_mb:.1f} MB) exceeds {provider} limit ({limit} MB)", verbose)

            if provider == "openai":
                result.update(process_with_openai_whisper(audio_path, model=model, verbose=verbose))
            elif provider == "groq":
                result.update(process_with_groq_whisper(audio_path, model=model, verbose=verbose))
            elif provider == "assemblyai":
                result.update(process_with_assemblyai(audio_path, model=model, verbose=verbose))
            elif provider == "deepgram":
                result.update(process_with_deepgram(audio_path, model=model, verbose=verbose))
            elif provider == "local":
                result.update(process_with_local_whisper(audio_path, model=model, verbose=verbose))
            else:
                raise ValueError(f"Unknown provider: {provider}")

    log("Done!", verbose)
    return result


def list_models():
    """Print available models for each provider."""
    providers = get_available_providers()

    print("Available models by provider:\n")

    for provider in providers["all_providers"]:
        models = AVAILABLE_MODELS.get(provider, [])
        default = DEFAULT_MODELS.get(provider, "")

        print(f"  {provider}:")
        for m in models:
            marker = " (default)" if m == default else ""
            print(f"    - {m}{marker}")
        print()

    if not providers["all_providers"]:
        print("  No providers available. Set API keys to enable providers.")


def main():
    parser = argparse.ArgumentParser(
        description="Process video for understanding/transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process YouTube video
  %(prog)s "https://youtube.com/watch?v=..."

  # Process local file
  %(prog)s video.mp4

  # Use specific provider and model
  %(prog)s video.mp4 --provider openrouter --model google/gemini-3-pro-preview

  # Custom prompt
  %(prog)s video.mp4 -p "List all products shown with timestamps"

  # ASR-only (transcription without visual analysis)
  %(prog)s video.mp4 --asr-only

  # List available models
  %(prog)s --list-models
        """
    )

    parser.add_argument("source", nargs="?", help="YouTube URL, video URL, or local file path")
    parser.add_argument("-p", "--prompt", help="Custom prompt for video understanding")
    parser.add_argument("--provider", help="Force specific provider (gemini, openrouter, openai, groq, assemblyai, deepgram, local)")
    parser.add_argument("-m", "--model", help="Force specific model (use --list-models to see options)")
    parser.add_argument("--asr-only", action="store_true", help="Force ASR-only mode (no visual analysis)")
    parser.add_argument("-o", "--output", help="Output JSON file (default: stdout)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress messages")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--list-providers", action="store_true", help="List available providers and exit")

    args = parser.parse_args()

    # Handle list commands
    if args.list_models:
        list_models()
        return

    if args.list_providers:
        providers = get_available_providers()
        print(json.dumps(providers, indent=2))
        return

    # Require source for processing
    if not args.source:
        parser.print_help()
        sys.exit(1)

    try:
        result = process_video(
            source=args.source,
            prompt=args.prompt,
            provider=args.provider,
            model=args.model,
            asr_only=args.asr_only,
            verbose=not args.quiet,
        )

        output = json.dumps(result, indent=2, ensure_ascii=False)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            print(output)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
