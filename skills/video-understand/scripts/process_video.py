#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main video processing script with automatic provider selection.
Handles YouTube URLs, local files, and other video URLs.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

# Import provider check
from check_providers import get_available_providers


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


def download_video(url: str, output_dir: str) -> str:
    """Download video using yt-dlp."""
    output_path = os.path.join(output_dir, "video.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", output_path,
        "--no-playlist",
        url
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # Find the downloaded file
    for f in os.listdir(output_dir):
        if f.startswith("video."):
            return os.path.join(output_dir, f)
    raise FileNotFoundError("Downloaded video not found")


def extract_audio(video_path: str, output_dir: str) -> str:
    """Extract audio from video using ffmpeg."""
    audio_path = os.path.join(output_dir, "audio.mp3")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",  # No video
        "-acodec", "libmp3lame",
        "-ab", "128k",
        "-ar", "16000",  # 16kHz for Whisper
        "-y",  # Overwrite
        audio_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return audio_path


def get_video_duration(path: str) -> float:
    """Get video/audio duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


# === Provider-specific processors ===

def process_with_gemini(source: str, prompt: str, is_url: bool = False) -> dict:
    """Process video with Google Gemini."""
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.0-flash")

    if is_url and is_youtube_url(source):
        # Gemini can handle YouTube URLs directly
        response = model.generate_content([
            prompt,
            {"video_url": source}
        ])
    else:
        # Upload local file
        video_file = genai.upload_file(source)

        # Wait for processing
        import time
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise RuntimeError(f"Video processing failed: {video_file.state.name}")

        response = model.generate_content([prompt, video_file])

    return {
        "provider": "gemini",
        "capability": "full_video",
        "response": response.text,
    }


def process_with_openrouter(video_path: str, prompt: str) -> dict:
    """Process video with OpenRouter (using Gemini models)."""
    import base64
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"]
    )

    # Read and encode video
    with open(video_path, "rb") as f:
        video_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    # Get mime type
    ext = Path(video_path).suffix.lower()
    mime_types = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}
    mime_type = mime_types.get(ext, "video/mp4")

    response = client.chat.completions.create(
        model="google/gemini-3-flash-preview",
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
        "capability": "full_video",
        "response": response.choices[0].message.content,
    }


def process_with_openai_whisper(audio_path: str) -> dict:
    """Transcribe audio with OpenAI Whisper API."""
    from openai import OpenAI

    client = OpenAI()

    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    segments = []
    if hasattr(transcript, 'segments'):
        for seg in transcript.segments:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            })

    return {
        "provider": "openai",
        "capability": "asr_only",
        "transcript": segments,
        "text": transcript.text,
        "language": getattr(transcript, 'language', None),
    }


def process_with_groq_whisper(audio_path: str) -> dict:
    """Transcribe audio with Groq Whisper."""
    from groq import Groq

    client = Groq()

    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
            response_format="verbose_json",
        )

    segments = []
    if hasattr(transcript, 'segments'):
        for seg in transcript.segments:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            })

    return {
        "provider": "groq",
        "capability": "asr_only",
        "transcript": segments,
        "text": transcript.text,
        "language": getattr(transcript, 'language', None),
    }


def process_with_assemblyai(audio_path: str) -> dict:
    """Transcribe audio with AssemblyAI."""
    import assemblyai as aai

    aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]

    config = aai.TranscriptionConfig(
        speaker_labels=True,
        auto_chapters=True,
    )

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    segments = []
    if transcript.utterances:
        for utt in transcript.utterances:
            segments.append({
                "start": utt.start / 1000,  # ms to seconds
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
        "capability": "asr_only",
        "transcript": segments,
        "text": transcript.text,
        "chapters": chapters,
    }


def process_with_deepgram(audio_path: str) -> dict:
    """Transcribe audio with Deepgram."""
    from deepgram import DeepgramClient, PrerecordedOptions, FileSource

    client = DeepgramClient(os.environ["DEEPGRAM_API_KEY"])

    with open(audio_path, "rb") as f:
        buffer_data = f.read()

    payload: FileSource = {"buffer": buffer_data}

    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        utterances=True,
        punctuate=True,
        diarize=True,
    )

    response = client.listen.prerecorded.v("1").transcribe_file(payload, options)
    result = response.to_dict()

    segments = []
    if "utterances" in result["results"]:
        for utt in result["results"]["utterances"]:
            segments.append({
                "start": utt["start"],
                "end": utt["end"],
                "text": utt["transcript"],
                "speaker": utt.get("speaker"),
            })

    return {
        "provider": "deepgram",
        "capability": "asr_only",
        "transcript": segments,
        "text": result["results"]["channels"][0]["alternatives"][0]["transcript"],
    }


def process_with_local_whisper(audio_path: str, model: str = "base") -> dict:
    """Transcribe audio with local Whisper."""
    try:
        import whisper

        model_obj = whisper.load_model(model)
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
            "capability": "asr_only",
            "transcript": segments,
            "text": result["text"],
            "language": result.get("language"),
        }
    except ImportError:
        # Fall back to CLI
        cmd = ["whisper", audio_path, "--model", model, "--output_format", "json"]
        subprocess.run(cmd, check=True)

        # Read output
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
            "capability": "asr_only",
            "transcript": segments,
            "text": result["text"],
        }


def process_video(
    source: str,
    prompt: str = "Provide a detailed transcript with timestamps. Also describe any important visual elements, on-screen text, and key moments.",
    provider: str = None,
    asr_only: bool = False,
) -> dict:
    """
    Process a video with automatic provider selection.

    Args:
        source: YouTube URL, local file path, or video URL
        prompt: Prompt for video understanding (used by Gemini/OpenRouter)
        provider: Force specific provider (optional)
        asr_only: Force ASR-only mode even if video understanding available

    Returns:
        dict with transcript, analysis, and metadata
    """

    # Check available providers
    providers = get_available_providers()

    if provider:
        if provider not in providers["all_providers"]:
            raise ValueError(f"Provider '{provider}' not available. Available: {providers['all_providers']}")
    elif asr_only:
        provider = providers["asr_only"][0] if providers["asr_only"] else None
    else:
        provider = providers["recommended"]

    if not provider:
        raise RuntimeError("No video understanding providers available")

    # Determine source type
    is_url = source.startswith(("http://", "https://"))
    is_youtube = is_url and is_youtube_url(source)
    is_local = not is_url and os.path.isfile(source)

    if not is_url and not is_local:
        raise FileNotFoundError(f"Source not found: {source}")

    result = {
        "source": {
            "type": "youtube" if is_youtube else ("url" if is_url else "local"),
            "path": source,
        }
    }

    # === Full Video Understanding ===
    if provider in providers["video_understanding"] and not asr_only:

        if provider == "gemini":
            if is_youtube:
                # Gemini handles YouTube directly
                result.update(process_with_gemini(source, prompt, is_url=True))
            elif is_local:
                result.update(process_with_gemini(source, prompt, is_url=False))
            else:
                # Download URL first
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = download_video(source, tmpdir)
                    result.update(process_with_gemini(video_path, prompt, is_url=False))

        elif provider == "openrouter":
            if is_local:
                result.update(process_with_openrouter(source, prompt))
            else:
                # Need to download for OpenRouter
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = download_video(source, tmpdir)
                    result.update(process_with_openrouter(video_path, prompt))

        elif provider == "vertex":
            # Similar to Gemini but with Vertex AI client
            # TODO: Implement Vertex AI specific handling
            raise NotImplementedError("Vertex AI support coming soon")

    # === ASR-Only Providers ===
    else:
        # Need to extract audio for ASR
        with tempfile.TemporaryDirectory() as tmpdir:
            if is_url:
                video_path = download_video(source, tmpdir)
            else:
                video_path = source

            audio_path = extract_audio(video_path, tmpdir)

            if provider == "openai":
                result.update(process_with_openai_whisper(audio_path))
            elif provider == "groq":
                result.update(process_with_groq_whisper(audio_path))
            elif provider == "assemblyai":
                result.update(process_with_assemblyai(audio_path))
            elif provider == "deepgram":
                result.update(process_with_deepgram(audio_path))
            elif provider == "local":
                result.update(process_with_local_whisper(audio_path))
            else:
                raise ValueError(f"Unknown provider: {provider}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Process video for understanding/transcription")
    parser.add_argument("source", help="YouTube URL, video URL, or local file path")
    parser.add_argument("-p", "--prompt", default=None, help="Custom prompt for video understanding")
    parser.add_argument("--provider", help="Force specific provider")
    parser.add_argument("--asr-only", action="store_true", help="Force ASR-only mode")
    parser.add_argument("-o", "--output", help="Output JSON file")

    args = parser.parse_args()

    prompt = args.prompt or "Provide a detailed transcript with timestamps. Describe important visual elements, on-screen text, and key moments."

    result = process_video(
        source=args.source,
        prompt=prompt,
        provider=args.provider,
        asr_only=args.asr_only,
    )

    output = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Output written to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
