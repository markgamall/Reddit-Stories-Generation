import os
import time
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import yt_dlp
import re
from moviepy import VideoFileClip
from pydub import AudioSegment
import whisper
from gridfs import GridFS
from bson.objectid import ObjectId

# Load the model once when the module is imported
default_model = "base.en"
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

        if output_name:
            file_path = f"{output_name}_youtube_transcription.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            print(f"YouTube transcription saved as: {file_path}")
        
        return transcript_text
    except Exception as e:
        print(f"Error fetching YouTube transcription: {e}")
        return None

def check_transcription_validity(transcription):
    patterns = [
        r"\[.*?\]", 
        r"\(.*?\)",  
    ]
    
    combined_pattern = f"^({'|'.join(patterns)})+$"
    
    if re.fullmatch(combined_pattern, transcription.strip()):
        return "not valid"
    else:
        return "valid"

def download_video(video_url, output_dir=".", fs=None):
    """Download video and store it in GridFS if available"""
    # Common options for both local and GridFS methods
    common_opts = {
        'format': 'best',
        'noplaylist': True,
        'postprocessors': [
            {'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'}
        ],
        # Add these options to bypass YouTube restrictions
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': False,  # Set to True in production
        'geo_bypass': True,
        'geo_bypass_country': 'US',
        'extractor_retries': 5,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
        }
    }

    if fs is None:
        # Original method (for local use without GridFS)
        opts = {**common_opts, 'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s')}

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info_dict = ydl.extract_info(video_url, download=True)
                video_title = sanitize_filename(info_dict.get('title', 'Unknown_Title'))
                print(f"Video downloaded as: {video_title}.mp4")
                return os.path.join(output_dir, f"{video_title}.mp4"), video_title
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None, None
    else:
        # GridFS method (for deployed environment)
        opts = {**common_opts, 'outtmpl': 'temp_video.%(ext)s'}

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info_dict = ydl.extract_info(video_url, download=True)
                video_title = sanitize_filename(info_dict.get('title', 'Unknown_Title'))
                temp_video_path = "temp_video.mp4"  # This will be the local temp file
                
                # Store the video in MongoDB GridFS
                with open(temp_video_path, 'rb') as video_file:
                    file_id = fs.put(video_file, filename=f"{video_title}.mp4")
                
                # Remove the local temp file
                os.remove(temp_video_path)
                
                print(f"Video downloaded and stored in MongoDB with ID: {file_id}")
                return str(file_id), video_title  # Return GridFS ID instead of file path
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None, None

def extract_audio_from_video(video_path, audio_trimmed_path, fs=None):
    """Extract audio from video file or GridFS ID"""
    if fs is None or not isinstance(video_path, str) or not video_path.strip() or os.path.exists(video_path):
        # Original method (for local use without GridFS or when path is a file)
        try:
            clip = VideoFileClip(video_path)
            audio_path = "output_audio.wav"
            clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(audio_trimmed_path, format="wav")

            print(f"Audio extracted and saved to: {audio_trimmed_path}")
            return audio_trimmed_path
        except Exception as e:
            print(f"Error extracting audio from video: {e}")
            return None
    else:
        # GridFS method (for deployed environment)
        try:
            # Retrieve the video from GridFS
            temp_video_path = "temp_retrieved_video.mp4"
            with open(temp_video_path, 'wb') as f:
                f.write(fs.get(ObjectId(video_path)).read())
            
            # Process the video as before
            clip = VideoFileClip(temp_video_path)
            audio_path = "output_audio.wav"
            clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(audio_trimmed_path, format="wav")
            
            # Store the audio in GridFS
            with open(audio_trimmed_path, 'rb') as audio_file:
                audio_file_id = fs.put(audio_file, filename="processed_audio.wav")
            
            # Clean up local temp files
            os.remove(temp_video_path)
            os.remove(audio_path)
            os.remove(audio_trimmed_path)
            
            # Remove the video from GridFS as we don't need it anymore
            fs.delete(ObjectId(video_path))
            
            print(f"Audio extracted and stored in MongoDB with ID: {audio_file_id}")
            return str(audio_file_id)  # Return GridFS ID instead of file path
        except Exception as e:
            print(f"Error extracting audio from video: {e}")
            return None

def generate_whisper_transcription(audio_path, fs=None):
    """Generate transcription with Whisper from audio file or GridFS ID"""
    if fs is None or not isinstance(audio_path, str) or not audio_path.strip() or os.path.exists(audio_path):
        # Original method (for local use without GridFS or when path is a file)
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
    else:
        # GridFS method (for deployed environment)
        try:
            # Retrieve the audio from GridFS
            temp_audio_path = "temp_retrieved_audio.wav"
            with open(temp_audio_path, 'wb') as f:
                f.write(fs.get(ObjectId(audio_path)).read())
            
            # Process with Whisper
            audio = whisper.load_audio(temp_audio_path)
            print("Audio loaded successfully for Whisper.")

            start_time = time.time()
            transcription = model.transcribe(audio)
            end_time = time.time()

            elapsed_time = end_time - start_time
            print(f"Time Taken by Whisper: {elapsed_time:.4f} seconds")
            
            # Clean up local temp file and GridFS stored audio
            os.remove(temp_audio_path)
            fs.delete(ObjectId(audio_path))
            
            return transcription["text"]
        except Exception as e:
            print(f"Error generating Whisper transcription: {e}")
            return None

def process_youtube_url(video_url, output_dir=".", fs=None):
    """Process a YouTube URL to get transcription using either YouTube API or Whisper"""
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
        video_path, video_title = download_video(video_url, output_dir, fs)
        
        if not video_path:
            return {"success": False, "error": "Failed to download video"}
        
        audio_trimmed_path = os.path.join(output_dir, "output_audio_trimmed.wav")
        audio_path = extract_audio_from_video(video_path, audio_trimmed_path, fs)
        
        if not audio_path:
            return {"success": False, "error": "Failed to extract audio from video"}
        
        transcript_text = generate_whisper_transcription(audio_path, fs)
        transcription_method = "Whisper"
        
        # Clean up temporary files (if not using GridFS)
        if fs is None:
            try:
                os.remove(video_path)
                os.remove(audio_path)
                os.remove("output_audio.wav")
            except:
                pass
    
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