import os
import time
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter
from pytube.exceptions import PytubeError
from pathlib import Path
from pytube import YouTube
from yt_dlp import YoutubeDL
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
    'ignoreerrors': True,  # Continue on download errors
    'geo_bypass': True,    # Try to bypass geo-restrictions
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

def get_video_transcription(video_id, output_name=None, retries=3, delay=5):
    for attempt in range(retries):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            logger.info(f"YouTube transcript fetched successfully on attempt {attempt + 1}")

            formatter = TextFormatter()
            transcript_text = formatter.format_transcript(transcript)

            if output_name and output_name != "":
                file_path = f"{output_name}_youtube_transcription.txt"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(transcript_text)
                logger.info(f"YouTube transcription saved as: {file_path}")
            
            return transcript_text
        
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                logger.info(f"Retrying after {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Failed after {retries} attempts: {e}")
                return None
        
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
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


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def download_audio_yt_dlp(video_url, output_dir):
    temp_filename = f"temp_audio_{int(time.time())}"
    output_template = os.path.join(output_dir, f"{temp_filename}.%(ext)s")

    opts = {
    'format': 'bestaudio/best',  # Get best audio format available
    'outtmpl': output_template,
    'noplaylist': True,
    'quiet': True,
    'no_warnings': True,
    'geo_bypass': True,  # Try to bypass geo-restrictions
    'nocheckcertificate': True,      # Skip SSL cert check (fixes some network issues)
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    }

    with YoutubeDL(opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        video_title = sanitize_filename(info_dict.get('title', 'Unknown_Title'))

        # Search for the audio file
        for file in os.listdir(output_dir):
            if file.startswith(temp_filename):
                return os.path.join(output_dir, file), video_title

        raise FileNotFoundError("Audio file was not found after yt-dlp download.")

def download_audio_pytube(video_url, output_dir):
    yt = YouTube(video_url)
    stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
    if not stream:
        raise ValueError("No audio stream found with pytube.")
    
    video_title = sanitize_filename(yt.title)
    output_path = stream.download(output_path=output_dir, filename=f"{video_title}.mp4")
    return output_path, video_title

def download_audio_directly(video_url, output_dir="."):
    """Download audio from a YouTube video with fallback from yt-dlp to pytube"""
    os.makedirs(output_dir, exist_ok=True)

    try:
        logger.info("Trying yt-dlp for audio download...")
        return download_audio_yt_dlp(video_url, output_dir)
    
    except Exception as e:
        logger.warning(f"yt-dlp failed: [{type(e).__name__}] {e}")
        logger.info("Falling back to pytube...")

        try:
            return download_audio_pytube(video_url, output_dir)
        except PytubeError as pe:
            logger.error(f"pytube also failed: [{type(pe).__name__}] {pe}")
        except Exception as e:
            logger.error(f"Unhandled error in pytube: [{type(e).__name__}] {e}")

    return None, None

def generate_whisper_transcription(audio_path=None):
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



def retry_with_backoff(func, max_attempts=3, initial_delay=1, *args, **kwargs):
    """Retries a function with exponential backoff."""
    delay = initial_delay
    for attempt in range(max_attempts):
        try:
            result = func(*args, **kwargs)
            if result:  # assuming None or False means failure
                return result
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed with error: {e}")
        time.sleep(delay)
        delay *= 2  # exponential backoff
    return None


def download_subtitles(video_url, retries=3, delay=5):
    """
    Tries to download subtitles for a given YouTube video using yt-dlp.
    Returns subtitle text if available, or None if not.
    """
    for attempt in range(retries):
        try:
            ydl_opts = {
                'writesubtitles': True,  # Enable subtitle download
                'subtitleslangs': ['en'],  # Subtitle language (English)
                'quiet': True,  # Suppress unnecessary output
                'outtmpl': '%(id)s_subtitles.%(ext)s',  # Output filename template
            }

            with YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(video_url, download=False)  # Get video info without downloading
                subtitles = info_dict.get('subtitles', {})
                
                if 'en' in subtitles:  # Check if subtitles are available in English
                    subtitle_file = f"{info_dict['id']}_en.vtt"  # VTT file containing subtitles
                    transcript_text = ""
                    # Extract subtitle text from .vtt file
                    with open(subtitle_file, 'r', encoding='utf-8') as f:
                        transcript_text = f.read()
                    return transcript_text
                else:
                    logger.warning(f"No English subtitles found for video: {video_url}")
                    return None

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed to download subtitles: {e}")
            if attempt < retries - 1:
                logger.info(f"Retrying after {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Failed to download subtitles after {retries} attempts: {e}")
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

    # Try multiple times to get video info
    video_info = retry_with_backoff(get_video_info, max_attempts=3, initial_delay=2, video_url=video_url)
    if not video_info:
        return {
            "success": False,
            "error": "YouTube content access restricted or temporarily unavailable. "
                     "This video may be region-blocked or protected. Retry later or use a VPN."
        }
    
    # Step 1: Try YouTube API transcription
    transcript_text = get_video_transcription(video_id)
    transcription_method = "YouTube API"
    
    if transcript_text:
        logger.info("YouTube transcription fetched successfully.")
    else:
        logger.info("YouTube API transcription not found or invalid. Falling back to yt-dlp subtitles.")
        
        # Step 2: If YouTube API transcription is not available, try to download subtitles with yt-dlp
        transcript_text = download_subtitles(video_url)
        transcription_method = "yt-dlp subtitles"
        
        if transcript_text:
            logger.info("Subtitles downloaded successfully using yt-dlp.")
        else:
            logger.info("No subtitles found. Falling back to Whisper transcription.")

            # Step 3: If no transcription or subtitles are available, use Whisper to transcribe audio
            audio_dir = os.path.join(output_dir, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            
            audio_result = retry_with_backoff(download_audio_directly, max_attempts=3, initial_delay=2, video_url=video_url, output_dir=audio_dir)
            if not audio_result:
                return {"success": False, "error": "Audio download failed after multiple attempts. The video may be restricted or unavailable."}
            audio_path, video_title = audio_result 
            
            if not audio_path:
                return {"success": False, "error": "Audio download failed. Video may be restricted or inaccessible."}
            
            transcript_text = retry_with_backoff(generate_whisper_transcription, max_attempts=3, initial_delay=2, audio_path=audio_path)
            transcription_method = "Whisper"
            
            if fs:
                try:
                    with open(audio_path, 'rb') as audio_file:
                        audio_file_id = fs.put(audio_file, filename=f"{video_title}_audio.wav")
                        logger.info(f"Audio file stored in GridFS with ID: {audio_file_id}")
                except Exception as e:
                    logger.error(f"Error storing audio in GridFS: {e}")
            
            # Clean up
            with contextlib.suppress(Exception):
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                os.rmdir(audio_dir)
                if output_dir.startswith(tempfile.gettempdir()):
                    os.rmdir(output_dir)

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