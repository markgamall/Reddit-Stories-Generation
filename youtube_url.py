import os
import time
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import yt_dlp
import re
import whisper
import contextlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model once when the module is imported
default_model = "tiny.en"
model = whisper.load_model(default_model)

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def get_video_id_from_url(video_url):
    if "watch?v=" in video_url:
        return video_url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in video_url:
        return video_url.split("youtu.be/")[1].split("?")[0]
    return None

def get_video_info(video_url):
    """Get title and other info without downloading the video"""
    opts = {
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            video_title = sanitize_filename(info_dict.get('title', 'Unknown_Title'))
            return {
                'title': video_title,
                'channel': info_dict.get('uploader', 'Unknown Channel'),
                'duration': info_dict.get('duration', 0),
                'view_count': info_dict.get('view_count', 0),
                'upload_date': info_dict.get('upload_date', 'Unknown')
            }
    except Exception as e:
        logger.error(f"Error fetching video info: {e}")
        return None

def get_video_transcription(video_id, output_name=None):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        logger.info("YouTube transcript fetched successfully.")

        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript)

        if output_name and output_name != "":
            file_path = f"{output_name}_youtube_transcription.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            logger.info(f"YouTube transcription saved as: {file_path}")
        
        return transcript_text
    except Exception as e:
        logger.error(f"Error fetching YouTube transcription: {e}")
        return None

def check_transcription_validity(transcription):
    if not transcription:
        return "not valid"
        
    patterns = [
        r"\[.*?\]", 
        r"\(.*?\)",  
    ]
    
    combined_pattern = f"^({'|'.join(patterns)})+$"
    
    if re.fullmatch(combined_pattern, transcription.strip()):
        return "not valid"
    else:
        return "valid"

def download_audio_directly(video_url, output_dir="."):
    """Download just the audio track using yt-dlp without requiring ffmpeg"""
    os.makedirs(output_dir, exist_ok=True)
    
    temp_filename = f"temp_audio_{int(time.time())}"
    output_template = os.path.join(output_dir, f"{temp_filename}.%(ext)s")
    
    opts = {
        'format': 'bestaudio',  # Get best audio
        'outtmpl': output_template,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        # No ffmpeg post-processors
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            video_title = sanitize_filename(info_dict.get('title', 'Unknown_Title'))
            
            # Find the downloaded file
            audio_path = None
            for file in os.listdir(output_dir):
                if file.startswith(temp_filename):
                    audio_path = os.path.join(output_dir, file)
                    break
                    
            if audio_path:
                logger.info(f"Audio downloaded to: {audio_path}")
                return audio_path, video_title
            else:
                logger.error("Could not locate downloaded audio file")
                return None, video_title
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        return None, None

def generate_whisper_transcription(audio_path):
    try:
        audio = whisper.load_audio(audio_path)
        logger.info("Audio loaded successfully for Whisper.")

        start_time = time.time()
        transcription = model.transcribe(audio)
        end_time = time.time()

        elapsed_time = end_time - start_time
        logger.info(f"Time Taken by Whisper: {elapsed_time:.4f} seconds")
        
        return transcription["text"]
    except Exception as e:
        logger.error(f"Error generating Whisper transcription: {e}")
        return None

def process_youtube_url(video_url, output_dir=None, fs=None):
    """Process a YouTube URL to get transcription using either YouTube API or Whisper.
    This version skips video download and extracts audio directly."""
    # Use a temporary directory if none specified
    if output_dir is None or output_dir == ".":
        temp_base_dir = tempfile.gettempdir()
        output_dir = os.path.join(temp_base_dir, f"youtube_process_{int(time.time())}")
        os.makedirs(output_dir, exist_ok=True)
    
    video_id = get_video_id_from_url(video_url)
    if not video_id:
        return {"success": False, "error": "Could not extract video ID from URL"}
    
    # Try to get info first
    video_info = get_video_info(video_url)
    if not video_info:
        return {"success": False, "error": "Could not fetch video information"}
    
    # Try to get YouTube transcription first
    transcript_text = get_video_transcription(video_id)
    transcription_method = "YouTube API"
    
    # If YouTube transcription fails or is invalid, use Whisper
    if not transcript_text or check_transcription_validity(transcript_text) == "not valid":
        logger.info("YouTube transcription not available or invalid. Using Whisper...")
        
        # Create specific directory for audio
        audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        # Download audio directly instead of downloading video then extracting audio
        audio_path, video_title = download_audio_directly(video_url, audio_dir)
        
        if not audio_path:
            return {"success": False, "error": "Failed to download audio"}
        
        transcript_text = generate_whisper_transcription(audio_path)
        transcription_method = "Whisper"
        
        # If using GridFS, store the audio file
        if fs:
            try:
                with open(audio_path, 'rb') as audio_file:
                    audio_file_id = fs.put(audio_file, filename=f"{video_title}_audio.wav")
                    logger.info(f"Audio file stored in GridFS with ID: {audio_file_id}")
            except Exception as e:
                logger.error(f"Error storing audio in GridFS: {e}")
        
        # Clean up temporary files
        try:
            with contextlib.suppress(Exception):
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    
            # Try to clean up the directories
            with contextlib.suppress(Exception):
                os.rmdir(audio_dir)
                # Only remove the output_dir if we created it as a temp dir
                if output_dir.startswith(tempfile.gettempdir()):
                    os.rmdir(output_dir)
        except Exception as e:
            logger.warning(f"Warning during cleanup: {e}")
    
    if not transcript_text:
        return {"success": False, "error": "Failed to generate transcription"}
    
    return {
        "success": True, 
        "transcription": transcript_text,
        "method": transcription_method,
        "video_info": video_info,
        "video_id": video_id
    }


# Only run this if executed directly (not when imported)
if __name__ == "__main__":
    video_url = 'https://www.youtube.com/watch?v=loYuxWSsLNc&ab_channel=Visme'
    result = process_youtube_url(video_url)
    
    if result["success"]:
        print(f"Transcription method: {result['method']}")
        print(f"Transcription (first 300 chars): {result['transcription'][:300]}...")
    else:
        print(f"Error: {result['error']}")