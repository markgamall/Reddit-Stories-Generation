import os
import time
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import yt_dlp
import re
from moviepy import VideoFileClip
from pydub import AudioSegment
import whisper
import contextlib

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
        print(f"Error fetching video info: {e}")
        return None

def get_video_transcription(video_id, output_name=None):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        print("YouTube transcript fetched successfully.")

        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript)

        if output_name and output_name != "":
            file_path = f"{output_name}_youtube_transcription.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            print(f"YouTube transcription saved as: {file_path}")
        
        return transcript_text
    except Exception as e:
        print(f"Error fetching YouTube transcription: {e}")
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

def download_video(video_url, output_dir="."):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a temporary filename that doesn't rely on the video title
    temp_filename = f"temp_video_{int(time.time())}"
    output_template = os.path.join(output_dir, f"{temp_filename}.%(ext)s")
    
    opts = {
        'format': 'best[height<=720]',  # Limit resolution to 720p to speed up download
        'outtmpl': output_template,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [
            {'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'}
        ]
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            video_title = sanitize_filename(info_dict.get('title', 'Unknown_Title'))
            print(f"Video downloaded with temp name: {temp_filename}.mp4")
            
            # The actual file path based on the output template
            video_path = os.path.join(output_dir, f"{temp_filename}.mp4")
            if not os.path.exists(video_path):
                # If the exact filename doesn't exist, try to find a matching file
                for file in os.listdir(output_dir):
                    if file.startswith(temp_filename) and file.endswith(".mp4"):
                        video_path = os.path.join(output_dir, file)
                        break
                        
            return video_path, video_title
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None, None

def extract_audio_from_video(video_path, audio_output_dir):
    """Extract audio from video with error handling and directory management"""
    os.makedirs(audio_output_dir, exist_ok=True)
    audio_path = os.path.join(audio_output_dir, "output_audio.wav")
    audio_trimmed_path = os.path.join(audio_output_dir, "output_audio_trimmed.wav")
    
    try:
        clip = None
        try:
            clip = VideoFileClip(video_path)
            # Try with new API first, then fall back to old API if it fails
            try:
                clip.audio.write_audiofile(audio_path, logger=None)
            except TypeError:
                # Fall back to older version API
                clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        finally:
            if clip:
                clip.close()

        # Convert to format suitable for Whisper
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(audio_trimmed_path, format="wav")

        print(f"Audio extracted and saved to: {audio_trimmed_path}")
        return audio_trimmed_path
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return None

def generate_whisper_transcription(audio_path):
    try:
        audio = whisper.load_audio(audio_path)
        print("Audio loaded successfully for Whisper.")

        start_time = time.time()
        transcription = model.transcribe(audio)
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"Time Taken by Whisper: {elapsed_time:.4f} seconds")
        
        return transcription["text"]
    except Exception as e:
        print(f"Error generating Whisper transcription: {e}")
        return None

def process_youtube_url(video_url, output_dir=None, fs=None):
    """Process a YouTube URL to get transcription using either YouTube API or Whisper"""
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
        print("YouTube transcription not available or invalid. Using Whisper...")
        
        # Create specific directories for this process
        video_dir = os.path.join(output_dir, "video")
        audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        
        video_path, video_title = download_video(video_url, video_dir)
        
        if not video_path:
            return {"success": False, "error": "Failed to download video"}
        
        audio_path = extract_audio_from_video(video_path, audio_dir)
        
        if not audio_path:
            return {"success": False, "error": "Failed to extract audio from video"}
        
        transcript_text = generate_whisper_transcription(audio_path)
        transcription_method = "Whisper"
        
        # If using GridFS, store the audio file
        if fs:
            try:
                with open(audio_path, 'rb') as audio_file:
                    audio_file_id = fs.put(audio_file, filename=f"{video_title}_audio.wav")
                    print(f"Audio file stored in GridFS with ID: {audio_file_id}")
            except Exception as e:
                print(f"Error storing audio in GridFS: {e}")
        
        # Clean up temporary files
        try:
            # Use contextlib to suppress errors during cleanup
            with contextlib.suppress(Exception):
                if os.path.exists(video_path):
                    os.remove(video_path)
                    
                for audio_file in [os.path.join(audio_dir, "output_audio.wav"), 
                                  os.path.join(audio_dir, "output_audio_trimmed.wav")]:
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
                        
            # Try to clean up the directories
            with contextlib.suppress(Exception):
                os.rmdir(video_dir)
                os.rmdir(audio_dir)
                # Only remove the output_dir if we created it as a temp dir
                if output_dir.startswith(tempfile.gettempdir()):
                    os.rmdir(output_dir)
        except Exception as e:
            print(f"Warning during cleanup: {e}")
    
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