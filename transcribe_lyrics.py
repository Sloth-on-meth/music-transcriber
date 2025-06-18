#!/usr/bin/env python3
"""
Music Lyrics Transcriber - Transcribes Dutch and English lyrics from audio files
Using OpenAI's Whisper model
"""

import argparse
import os
import sys
import torch
import whisper
from tqdm import tqdm

def check_requirements():
    """Check if system meets requirements for running Whisper"""
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Processing will be slower on CPU.")
    else:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Check if ffmpeg is installed
    if os.system("ffmpeg -version > /dev/null 2>&1") != 0:
        print("Error: ffmpeg not found. Please install ffmpeg.")
        sys.exit(1)

def transcribe_audio(audio_path, model_name="medium", language=None):
    """
    Transcribe audio file using Whisper
    
    Parameters:
    - audio_path: Path to the audio file
    - model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
    - language: Language code (e.g., "en", "nl", or None for auto-detection)
    
    Returns:
    - Dictionary containing transcription results
    """
    print(f"Loading Whisper {model_name} model...")
    model = whisper.load_model(model_name)
    
    print(f"Transcribing {audio_path}...")
    transcribe_options = {"task": "transcribe"}
    
    # Add language if specified
    if language:
        transcribe_options["language"] = language
    
    result = model.transcribe(audio_path, **transcribe_options)
    return result

def format_output(result, output_format="text"):
    """
    Format the transcription output
    
    Parameters:
    - result: Whisper transcription result
    - output_format: "text" or "srt"
    
    Returns:
    - Formatted output as string
    """
    if output_format == "text":
        return result["text"]
    elif output_format == "srt":
        srt_content = ""
        for i, segment in enumerate(result["segments"], 1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()
            srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
        return srt_content
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def save_output(content, output_path):
    """Save content to file"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe Dutch and English lyrics from audio files")
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument(
        "-m", "--model", 
        choices=["tiny", "base", "small", "medium", "large"], 
        default="medium",
        help="Whisper model size (default: medium)"
    )
    parser.add_argument(
        "-l", "--language", 
        choices=["en", "nl", "auto"], 
        default="auto",
        help="Language of the audio (en=English, nl=Dutch, auto=auto-detect)"
    )
    parser.add_argument(
        "-f", "--format", 
        choices=["text", "srt"], 
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "-o", "--output", 
        help="Output file path (default: input filename with .txt or .srt extension)"
    )
    parser.add_argument(
        "--bilingual", 
        action="store_true",
        help="Optimize for bilingual content (Dutch-English)"
    )
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.isfile(args.audio_path):
        print(f"Error: File not found: {args.audio_path}")
        sys.exit(1)
    
    # Check system requirements
    check_requirements()
    
    # Set language option for Whisper
    language = None if args.language == "auto" else args.language
    
    # Set default output path if not specified
    if not args.output:
        base_name = os.path.splitext(args.audio_path)[0]
        extension = ".srt" if args.format == "srt" else ".txt"
        args.output = base_name + extension
    
    # Transcribe audio
    result = transcribe_audio(args.audio_path, args.model, language)
    
    # Format and save output
    output_content = format_output(result, args.format)
    save_output(output_content, args.output)
    
    # Print summary
    print("\nTranscription Summary:")
    print(f"  Duration: {result['segments'][-1]['end']:.2f} seconds")
    print(f"  Detected language: {result['language']} ({whisper.tokenizer.LANGUAGES.get(result['language'], 'Unknown')})")
    print(f"  Model used: {args.model}")
    print(f"  Format: {args.format}")
    if args.bilingual:
        print("  Optimized for Dutch-English bilingual content")

if __name__ == "__main__":
    main()
