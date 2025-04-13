import os
import math
import subprocess
import tempfile
import shutil
import json
from typing import List, Dict, Tuple
from openai import OpenAI  # Updated import
from pydub import AudioSegment
import numpy as np
from transformers import pipeline

class VideoMontageGenerator:
    def __init__(self):
        self.temp_dir = "temp_montage"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.openai_client = OpenAI()  # Initialize OpenAI client
        
    def clean_temp_files(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir, exist_ok=True)

    def get_video_duration(self, video_path: str) -> float:
        """Get duration of a video file"""
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return float(result.stdout)

    def analyze_text_segments(self, story_text: str, prompts: List[str]) -> List[Dict]:
        """Analyze text to match with visual prompts using OpenAI"""
        segments = []
        chunk_size = len(story_text) // len(prompts)
        
        for i, prompt in enumerate(prompts):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < len(prompts) - 1 else len(story_text)
            text_chunk = story_text[start_idx:end_idx]
            
            # Use OpenAI with structured prompt
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system",
                    "content": """You are analyzing text segments to match with visual prompts.
                    Respond with a JSON object containing:
                    1. score: A float between 0 and 1 indicating relevance
                    2. explanation: A brief explanation of the score
                    Example: {"score": 0.8, "explanation": "Strong match due to..."}"""
                }, {
                    "role": "user",
                    "content": f"Text segment:\n{text_chunk}\n\nVisual prompt:\n{prompt}"
                }]
            )
            
            try:
                # Parse the response as JSON
                result = json.loads(response.choices[0].message.content)
                relevance_score = float(result['score'])
                # Ensure score is between 0 and 1
                relevance_score = max(0.0, min(1.0, relevance_score))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Fallback to default score if parsing fails
                print(f"Warning: Failed to parse relevance score: {e}")
                relevance_score = 0.5
            
            segments.append({
                "text": text_chunk,
                "prompt": prompt,
                "relevance": relevance_score,
                "start_char": start_idx,
                "end_char": end_idx
            })
        
        return segments

    def create_montage(self, 
                      story_text: str,
                      audio_path: str,
                      video_paths: List[str],
                      image_paths: List[str],
                      prompts: List[str],
                      output_path: str) -> str:
        """Create final video montage with synchronized audio and visuals"""
        try:
            # Clean up temp files
            self.clean_temp_files()
            
            # Load and analyze audio duration
            audio = AudioSegment.from_mp3(audio_path)
            total_duration = len(audio) / 1000.0  # Convert to seconds
            
            # Analyze text segments with visuals
            all_visuals = video_paths + image_paths
            visual_segments = self.analyze_text_segments(story_text, prompts)
            
            # Calculate timing for each segment
            segment_durations = []
            current_time = 0
            
            for segment in visual_segments:
                # Calculate duration based on text length proportion
                segment_length = segment['end_char'] - segment['start_char']
                duration = (segment_length / len(story_text)) * total_duration
                segment_durations.append({
                    "start": current_time,
                    "duration": duration
                })
                current_time += duration
            
            # Prepare video parts
            video_parts = []
            
            for i, (visual_path, segment) in enumerate(zip(all_visuals, segment_durations)):
                output_segment = f"{self.temp_dir}/segment_{i}.mp4"
                
                if visual_path in video_paths:
                    # Handle video segments - no looping
                    video_duration = self.get_video_duration(visual_path)
                    if video_duration < segment["duration"]:
                        # Instead of looping, adjust the segment duration to match video length
                        self._copy_video(visual_path, output_segment)
                        # Update segment duration to match video length
                        segment["duration"] = video_duration
                    else:
                        # Trim video if longer than segment
                        self._trim_video(visual_path, output_segment, segment["duration"])
                else:
                    # Convert image to video segment - keep existing image handling
                    self._image_to_video(visual_path, output_segment, segment["duration"])
                
                video_parts.append(output_segment)
            
            # Concatenate all parts
            self._concatenate_videos(video_parts, audio_path, output_path)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error in video montage creation: {str(e)}")
        finally:
            self.clean_temp_files()

    def _copy_video(self, input_path: str, output_path: str):
        """Copy video without modification"""
        copy_cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        subprocess.run(copy_cmd, check=True)

    def _trim_video(self, input_path: str, output_path: str, target_duration: float):
        """Trim video to target duration"""
        trim_cmd = [
            'ffmpeg', '-y', '-i', input_path, '-t', str(target_duration),
            '-c:v', 'libx264', '-preset', 'medium', '-pix_fmt', 'yuv420p',
            output_path
        ]
        subprocess.run(trim_cmd, check=True)

    def _image_to_video(self, input_path: str, output_path: str, duration: float):
        """Convert image to video segment with specified duration and add subtle zoom effect"""
        # Calculate zoom parameters
        zoom_factor = 1.05  # Subtle 5% zoom
        
        # Create zooming effect filter
        zoom_filter = (
            f"zoompan=z='min(zoom+0.0015,{zoom_factor})':"  # Gradual zoom from 1 to 1.05
            f"d={int(duration*60)}:"  # Duration in frames (60fps)
            "x='iw/2-(iw/zoom/2)':"  # Center horizontally
            "y='ih/2-(ih/zoom/2)':"  # Center vertically
            "fps=60"  # 60fps for smooth motion
        )
        
        # Combine with scaling and padding
        filter_complex = (
            f"{zoom_filter},"
            "scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
        )
        
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i', input_path,
            '-t', str(duration),
            '-vf', filter_complex,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-tune', 'stillimage',
            '-pix_fmt', 'yuv420p',
            '-r', '60',  # 60fps output
            output_path
        ]
        subprocess.run(cmd, check=True)

    def _concatenate_videos(self, video_parts: List[str], audio_path: str, output_path: str):
        """Concatenate video parts and add audio"""
        try:
            # First concatenate videos without audio
            concat_file = f"{self.temp_dir}/concat.txt"
            temp_video = f"{self.temp_dir}/temp_concat.mp4"
            
            # Create concat file
            with open(concat_file, 'w', encoding='utf-8') as f:
                for part in video_parts:
                    f.write(f"file '{os.path.abspath(part)}'\n")
            
            # Concatenate videos with consistent encoding
            concat_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-pix_fmt', 'yuv420p',
                temp_video
            ]
            subprocess.run(concat_cmd, check=True)
            
            # Add audio to the concatenated video
            final_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-map', '0:v:0',
                '-map', '1:a:0',
                output_path
            ]
            subprocess.run(final_cmd, check=True)
            
            # Clean up temporary concatenated video
            if os.path.exists(temp_video):
                os.remove(temp_video)
                
        except subprocess.CalledProcessError as e:
            raise Exception(f"FFmpeg error during concatenation: {e.stderr if hasattr(e, 'stderr') else str(e)}")
        except Exception as e:
            raise Exception(f"Error in video concatenation: {str(e)}")  