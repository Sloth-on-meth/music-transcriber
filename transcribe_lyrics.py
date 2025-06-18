#!/usr/bin/env python3
"""
Music Lyrics Transcriber - Transcribes Dutch and English lyrics from audio files
Using OpenAI's Whisper model (local or cloud API)
"""

import argparse
import os
import sys
import json
import torch
import whisper
from tqdm import tqdm
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

# For cloud API
import requests

def load_config():
    """Load configuration from config.json"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: config.json not found at {config_path}")
        print("Please create a config.json file with your OpenAI API key:")
        print('{\n  "openai_api_key": "your-api-key-here"\n}')
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: config.json has invalid JSON format")
        sys.exit(1)

def check_requirements(use_cloud):
    """Check if system meets requirements for running Whisper"""
    if not use_cloud:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. Processing will be slower on CPU.")
        else:
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        
        # Check if ffmpeg is installed
        if os.system("ffmpeg -version > /dev/null 2>&1") != 0:
            print("Error: ffmpeg not found. Please install ffmpeg.")
            sys.exit(1)
    else:
        print("Using OpenAI Whisper API (cloud mode)")

def split_audio(audio_path, max_chunk_size=24*60, verbose=False):
    """
    Split a large audio file into chunks of specified length in seconds
    
    Parameters:
    - audio_path: Path to the audio file
    - max_chunk_size: Maximum chunk size in seconds (default: 24 minutes)
    - verbose: Whether to show detailed output
    
    Returns:
    - List of temporary files containing audio chunks
    """
    try:
        # Get audio info
        data, samplerate = sf.read(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        print("Make sure ffmpeg is properly installed and the audio file is valid.")
        sys.exit(1)
    
    # Get audio duration in seconds
    duration = len(data) / samplerate
    
    # If audio file is shorter than max_chunk_size, no need to split
    if duration <= max_chunk_size:
        if verbose:
            print(f"Audio duration: {duration:.2f} seconds, no need to split.")
        return [audio_path]
    
    if verbose:
        print(f"Audio duration: {duration:.2f} seconds, splitting into chunks of {max_chunk_size} seconds.")
    
    # Calculate number of chunks
    n_chunks = int(np.ceil(duration / max_chunk_size))
    
    # Create temporary files for chunks
    tmp_files = []
    
    # Calculate samples per chunk
    samples_per_chunk = int(max_chunk_size * samplerate)
    
    # Split audio into chunks
    for i in range(n_chunks):
        start_sample = i * samples_per_chunk
        end_sample = min((i + 1) * samples_per_chunk, len(data))
        
        chunk_data = data[start_sample:end_sample]
        
        # Create temporary file for chunk
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_file.name, chunk_data, samplerate)
        tmp_files.append(tmp_file.name)
        
        if verbose:
            start_time = start_sample / samplerate
            end_time = end_sample / samplerate
            print(f"  Chunk {i+1}/{n_chunks}: {start_time:.2f}s - {end_time:.2f}s -> {tmp_file.name}")
    
    return tmp_files

def merge_transcription_results(results, verbose=False):
    """
    Merge multiple transcription results into one
    
    Parameters:
    - results: List of transcription results
    - verbose: Whether to show detailed output
    
    Returns:
    - Merged transcription result
    """
    if len(results) == 1:
        return results[0]
    
    if verbose:
        print(f"Merging {len(results)} transcription results...")
    
    # Initialize merged result
    merged_result = {
        "text": "",
        "segments": [],
        "language": results[0]["language"]  # Use language from first result
    }
    
    # Merge texts and segments
    time_offset = 0
    for i, result in enumerate(results):
        # Add text with proper spacing
        if merged_result["text"] and result["text"]:
            merged_result["text"] += " " + result["text"]
        else:
            merged_result["text"] += result["text"]
        
        # Adjust segment timestamps and add to merged result
        for segment in result["segments"]:
            adjusted_segment = segment.copy()
            adjusted_segment["start"] += time_offset
            adjusted_segment["end"] += time_offset
            merged_result["segments"].append(adjusted_segment)
        
        # Update time offset for next chunk
        if result["segments"]:
            chunk_duration = result["segments"][-1]["end"]
            time_offset += chunk_duration
    
    return merged_result

def transcribe_audio_local(audio_path, model_name="medium", language=None, prompt=None, verbose=False):
    """
    Transcribe audio file using local Whisper model
    
    Parameters:
    - audio_path: Path to the audio file
    - model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
    - language: Language code (e.g., "en", "nl", or None for auto-detection)
    - prompt: Optional prompt to guide the transcription
    - verbose: Whether to show detailed output
    
    Returns:
    - Dictionary containing transcription results
    """
    print(f"Loading Whisper {model_name} model...")
    model = whisper.load_model(model_name)
    
    # Set fp16 based on CUDA availability (better performance on GPU)
    fp16 = torch.cuda.is_available()
    
    # Split audio into manageable chunks if needed
    audio_chunks = split_audio(audio_path, max_chunk_size=24*60, verbose=verbose)
    
    # Transcribe each chunk
    results = []
    for i, chunk_path in enumerate(audio_chunks):
        if len(audio_chunks) > 1:
            print(f"Transcribing chunk {i+1}/{len(audio_chunks)}...")
        else:
            print(f"Transcribing {audio_path}...")
        
        transcribe_options = {
            "task": "transcribe",
            "fp16": fp16,
            # Increase beam_size for higher accuracy
            "beam_size": 5,
            # No length limit to handle entire songs
            "max_length": None,
            # Condition on previous text for better continuity between chunks
            "initial_prompt": prompt or "",
        }
        
        # Add language if specified
        if language:
            transcribe_options["language"] = language
        
        # Add previous segment text to prompt for continuity between chunks
        if i > 0 and results:
            last_segments = results[i-1]["segments"][-3:]
            context = " ".join([seg["text"] for seg in last_segments])
            if prompt:
                transcribe_options["initial_prompt"] = f"{prompt}. {context}"
            else:
                transcribe_options["initial_prompt"] = context
        elif prompt:
            transcribe_options["initial_prompt"] = prompt
            if verbose:
                print(f"Using custom prompt: {prompt}")
        
        result = model.transcribe(chunk_path, **transcribe_options)
        results.append(result)
        
        # Clean up temporary files if they were created
        if chunk_path != audio_path:
            os.unlink(chunk_path)
    
    # Merge results if needed
    if len(results) > 1:
        return merge_transcription_results(results, verbose=verbose)
    else:
        return results[0]

def transcribe_audio_cloud(audio_path, api_key, language=None, prompt=None, verbose=False):
    """
    Transcribe audio file using OpenAI Whisper API
    
    Parameters:
    - audio_path: Path to the audio file
    - api_key: OpenAI API key
    - language: Language code (e.g., "en", "nl", or None for auto-detection)
    - prompt: Optional prompt to guide the transcription
    - verbose: Whether to show detailed output
    
    Returns:
    - Dictionary containing transcription results
    """
    print(f"Preparing to transcribe {audio_path} using OpenAI Whisper API...")
    
    # Check file size (API limit is 25 MB)
    file_size = os.path.getsize(audio_path) / (1024 * 1024)  # in MB
    if file_size > 25:
        print(f"Warning: File size ({file_size:.2f} MB) exceeds OpenAI API limit (25 MB)")
        print("Splitting audio into smaller chunks and processing sequentially...")
        
        # Split audio into chunks smaller than 25 MB
        audio_chunks = split_audio(audio_path, max_chunk_size=10*60, verbose=verbose)  # 10 minutes per chunk
        
        # Process each chunk and merge results
        results = []
        for i, chunk_path in enumerate(audio_chunks):
            if len(audio_chunks) > 1:
                print(f"Processing chunk {i+1}/{len(audio_chunks)} with API...")
            
            # Update prompt for continuity between chunks
            current_prompt = prompt
            if i > 0 and results:
                # Include previous text for context
                last_segments = results[-1]["segments"][-2:]
                context = " ".join([seg["text"] for seg in last_segments])
                if prompt:
                    current_prompt = f"{prompt}. {context}"
                else:
                    current_prompt = context
            
            # Process this chunk
            chunk_result = process_single_audio_chunk_with_api(
                chunk_path, api_key, language, current_prompt, verbose)
            results.append(chunk_result)
            
            # Clean up temporary file if created
            if chunk_path != audio_path:
                os.unlink(chunk_path)
        
        # Merge results
        if len(results) > 1:
            return merge_transcription_results(results, verbose=verbose)
        else:
            return results[0]
    else:
        # Process single file directly
        return process_single_audio_chunk_with_api(audio_path, api_key, language, prompt, verbose)

def process_single_audio_chunk_with_api(audio_path, api_key, language=None, prompt=None, verbose=False):
    """
    Process a single audio chunk with the OpenAI Whisper API
    
    Parameters:
    - audio_path: Path to the audio file
    - api_key: OpenAI API key
    - language: Language code
    - prompt: Optional prompt
    - verbose: Whether to show detailed output
    
    Returns:
    - Transcription result
    """
    if verbose:
        print(f"Sending {audio_path} to Whisper API...")
    
    # Prepare API request
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    url = "https://api.openai.com/v1/audio/transcriptions"
    
    # Set response format to get word-level timestamps
    response_format = "verbose_json"
    
    # Prepare form data
    with open(audio_path, "rb") as f:
        files = {
            "file": (os.path.basename(audio_path), f, "audio/mpeg"),
            "model": (None, "whisper-1"),
            "response_format": (None, response_format),
        }
        
        # Add language if specified
        if language:
            files["language"] = (None, language)
        
        # Add prompt if specified
        if prompt:
            files["prompt"] = (None, prompt)
            if verbose:
                print(f"Using custom prompt: {prompt}")
        
        if verbose:
            print("Sending request to OpenAI API...")
        
        # Make API request
        try:
            response = requests.post(url, headers=headers, files=files)
            response.raise_for_status()
            result = response.json()
            
            if verbose:
                print("API response received.")
            
            # Process segments or create them if not provided
            segments = []
            if "segments" in result:
                segments = result["segments"]
            else:
                # If API doesn't return segments, create a single segment
                segments = [{
                    "id": 0,
                    "start": 0,
                    "end": 0,  # We don't have duration info
                    "text": result.get("text", "")
                }]
            
            # Extract duration if available
            duration = None
            if "duration" in result:
                duration = result["duration"]
            elif segments and len(segments) > 0 and "end" in segments[-1]:
                duration = segments[-1]["end"]
            
            return {
                "text": result.get("text", ""),
                "segments": segments,
                "language": language or "auto-detected",
                "duration": duration
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error: API request failed: {e}")
            if hasattr(e, "response") and hasattr(e.response, "text"):
                print(f"API response: {e.response.text}")
            sys.exit(1)

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
        help="Whisper model size (default: medium, ignored when --cloud is used)"
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
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Use OpenAI Whisper API instead of local model (requires API key in config.json)"
    )
    parser.add_argument(
        "--prompt",
        help="Provide a prompt to guide the transcription"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show more detailed output during transcription"
    )
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.isfile(args.audio_path):
        print(f"Error: File not found: {args.audio_path}")
        sys.exit(1)
    
    # Check system requirements
    check_requirements(args.cloud)
    
    # Set language option for Whisper
    language = None if args.language == "auto" else args.language
    
    # Set default output path if not specified
    if not args.output:
        base_name = os.path.splitext(args.audio_path)[0]
        extension = ".srt" if args.format == "srt" else ".txt"
        args.output = base_name + extension
    
    # Prepare bilingual prompt if bilingual option is selected and no custom prompt
    prompt = args.prompt
    if args.bilingual and not prompt:
        prompt = "This is a bilingual song with Dutch and English lyrics."
    
    # Transcribe audio
    if args.cloud:
        # Load API key from config.json
        config = load_config()
        api_key = config.get("openai_api_key")
        if not api_key:
            print("Error: No OpenAI API key found in config.json")
            sys.exit(1)
        
        result = transcribe_audio_cloud(args.audio_path, api_key, language, prompt, verbose=args.verbose)
    else:
        result = transcribe_audio_local(args.audio_path, args.model, language, prompt, verbose=args.verbose)
    
    # Format and save output
    output_content = format_output(result, args.format)
    save_output(output_content, args.output)
    
    # Print summary
    print("\nTranscription Summary:")
    if result["segments"] and len(result["segments"]) > 0:
        print(f"  Duration: {result['segments'][-1]['end']:.2f} seconds")
    print(f"  Detected language: {result.get('language', 'Unknown')} ({whisper.tokenizer.LANGUAGES.get(result.get('language', ''), 'Unknown')})")
    if not args.cloud:
        print(f"  Model used: {args.model}")
    else:
        print("  Model used: Whisper API (cloud)")
    print(f"  Format: {args.format}")
    if args.prompt:
        print(f"  Custom prompt: {args.prompt}")
    elif args.bilingual:
        print("  Optimized for Dutch-English bilingual content")

if __name__ == "__main__":
    main()
