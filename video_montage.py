import os
import math
import subprocess
import tempfile
import shutil
import json
import logging
from typing import List, Dict, Tuple
from openai import OpenAI
from pydub import AudioSegment
import numpy as np
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoMontageGenerator:
    def __init__(self):
        self.temp_dir = "temp_montage"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.openai_client = OpenAI()
        
    def clean_temp_files(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir, exist_ok=True)

    def get_video_duration(self, video_path: str) -> float:
        """Get precise duration of a video file with error handling"""
        try:
            if not os.path.exists(video_path):
                logger.error(f"Video file does not exist: {video_path}")
                return 0.0
                
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"FFprobe error: {result.stderr}")
                return 0.0
                
            try:
                duration = float(result.stdout.strip())
                logger.info(f"Video duration for {os.path.basename(video_path)}: {duration:.3f} seconds")
                return duration
            except ValueError:
                logger.error(f"Could not parse duration output: '{result.stdout}'")
                return 0.0
                
        except Exception as e: 
            logger.error(f"Error getting video duration: {str(e)}")
            return 0.0  # Return 0 duration on error

    def analyze_text_segments(self, story_text: str, first_k_sentences: List[str], prompts: List[str], audio_duration: float, audio_path: str = None) -> List[Dict]:
        """Analyze text segments for perfect audio-visual sync using ChatGPT for intelligent chunking.
        First k sentences are assigned to videos, and the rest are intelligently chunked for images based on relevance.
        """
        import re
        import json
        
        # Calculate timing information for each word
        word_timings = []
        word_to_time = {}
        
        # Try to use forced alignment if audio path is provided
        if audio_path is not None:
            try:
                from transformers import pipeline
                asr = pipeline('automatic-speech-recognition', model='openai/whisper-large-v2', device=-1, return_timestamps='word')
                result = asr(audio_path)
                for chunk in result['chunks']:
                    word = chunk['text'].strip()
                    start = chunk['timestamp'][0]
                    end = chunk['timestamp'][1]
                    word_timings.append((word, start, end-start))
                    word_to_time[word] = (start, end)
            except Exception as e:
                logger.warning(f"ASR forced alignment failed: {e}, falling back to estimation.")
        
        # If forced alignment failed or wasn't provided, estimate word timings
        if not word_timings:
            words = story_text.split()
            total_words = len(words)
            avg_word_duration = audio_duration / total_words
            cumulative_time = 0.0
            for word in words:
                word_time = avg_word_duration * (0.8 + (0.4 * len(word) / 10))
                word_timings.append((word, cumulative_time, word_time))
                cumulative_time += word_time
            scaling_factor = audio_duration / cumulative_time
            word_timings = [(word, start * scaling_factor, duration * scaling_factor) for word, start, duration in word_timings]
        
        # Get number of video and image prompts
        video_segment_count = len(first_k_sentences)
        image_segment_count = len(prompts) - video_segment_count
        
        logger.info(f"Total prompts: {len(prompts)} (Video: {video_segment_count}, Image: {image_segment_count})")
        
        # Use ChatGPT to intelligently chunk the story based on image prompts
        chunks = self._chunk_story_with_chatgpt(story_text, first_k_sentences, prompts)
        
        # Initialize tracking variables
        current_position = 0
        current_word_index = 0
        group_segments = []
        
        # Process each chunk
        for chunk in chunks:
            chunk_text = chunk["chunk"]
            prompt = chunk["image"]
            is_video = False
            
            # Check if this is a video prompt (one of the first k sentences)
            for i, video_prompt in enumerate(prompts[:video_segment_count]):
                if prompt == video_prompt:
                    is_video = True
                    break
            
            # Calculate timing for this chunk
            chunk_words = chunk_text.split()
            word_count = len(chunk_words)
            
            # Find the position of this chunk in the story
            chunk_start_pos = story_text.find(chunk_text, current_position)
            if chunk_start_pos == -1:  # If exact match not found, try a more flexible approach
                # Try to find the first few words
                first_few_words = ' '.join(chunk_words[:min(10, len(chunk_words))])
                chunk_start_pos = story_text.find(first_few_words, current_position)
                if chunk_start_pos == -1:
                    chunk_start_pos = current_position  # Fallback
            
            # Count words before this chunk to determine timing
            words_before_chunk = len(story_text[:chunk_start_pos].split())
            start_word_idx = max(0, min(words_before_chunk, len(word_timings) - 1))
            end_word_idx = min(start_word_idx + word_count - 1, len(word_timings) - 1)
            
            # Get timing information
            start_time = word_timings[start_word_idx][1] if start_word_idx < len(word_timings) else 0
            if end_word_idx < len(word_timings):
                end_time = word_timings[end_word_idx][1] + word_timings[end_word_idx][2]
            else:
                end_time = word_timings[-1][1] + word_timings[-1][2]
                
            actual_duration = end_time - start_time
            
            # Add segment
            group_segments.append({
                "text": chunk_text,
                "prompt": prompt,
                "is_video": is_video,
                "duration": actual_duration,
                "actual_audio_duration": actual_duration,
                "start_time": start_time,
                "end_time": end_time,
                "start_char": chunk_start_pos,
                "end_char": chunk_start_pos + len(chunk_text),
                "word_count": word_count
            })
            
            current_position = chunk_start_pos + len(chunk_text)
        
        # Log final segment count
        video_segments = sum(1 for segment in group_segments if segment.get("is_video", False))
        image_segments = len(group_segments) - video_segments
        logger.info(f"Created {len(group_segments)} segments (Video: {video_segments}, Image: {image_segments})")
        
        return group_segments
        
    def _chunk_story_with_chatgpt(self, story_text: str, first_k_sentences: List[str], prompts: List[str]) -> List[Dict]:
        """Use ChatGPT to intelligently chunk the story based on relevance to image prompts.
        First k sentences are assigned to videos, and the rest are chunked for images.
        """
        import re
        import json
        
        # Split story into sentences for the first k sentences (for videos)
        all_sentences = re.split(r'(?<=[.!?])\s+', story_text.strip())
        all_sentences = [s for s in all_sentences if s.strip()]
        
        # Get video and image prompts
        video_prompt_count = len(first_k_sentences)
        video_prompts = prompts[:video_prompt_count]
        image_prompts = prompts[video_prompt_count:]
        
        # Create initial chunks for videos (first k sentences)
        chunks = []
        for i, prompt in enumerate(video_prompts):
            if i < len(all_sentences):
                chunks.append({
                    "chunk": all_sentences[i],
                    "image": prompt
                })
        
        # If no image prompts, return just the video chunks
        if not image_prompts:
            return chunks
        
        # Prepare the remaining story text for image chunking
        remaining_story = ' '.join(all_sentences[video_prompt_count:])
        
        # Use ChatGPT to chunk the remaining story based on image prompts
        system_message = """
        You are a professional story chunker and JSON formatter.
        
        Your task is:
        
        1. You will receive a story text along with a list of image generation prompts.
        
        2. Your goal is to chunk the story based on the following exact rules:
        
        - Carefully read and match the sentences to the image generation prompts.
        
        - Whenever you find a sentence that is **very relevant** or **similar** to an image generation prompt:
        
        - Split a new chunk starting from that sentence.
        
        - Keep that sentence and all sentences that come after it together in the same chunk, **until** you find another sentence that is highly relevant to the next image generation prompt.
        
        - When you find another matching sentence, split again into a new chunk.
        
        - Continue this process until the end of the story.
        
        3. Very important rules:
        
        - **NEVER miss**, **remove**, **reorder**, or **modify** any part of the story text. All the original text must be present exactly as it is.
        
        - **NEVER ignore any image prompt**. Each image prompt must be matched to a chunk of the story text.
        
        - Only chunk the story into parts based on the logic above.
        
        - Each JSON element must contain:
        
        - A "chunk" field: The exact text of that part of the story.
        
        - An "image" field: The associated image prompt (copy the full prompt text exactly).
        
        4. Format your final output as a JSON list where each element has two fields: "chunk" and "image".
        
        Be extremely careful and precise.
        """
        
        try:
            # Call ChatGPT to chunk the story
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Story text: {remaining_story}\n\nImage prompts: {json.dumps(image_prompts)}"}
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.2,
                max_tokens=4000
            )
            
            # Extract and parse the response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)  
            
            # Parse the JSON responseee
            image_chunks = json.loads(response_text)
            
            # Merge chunks with the same image prompt
            image_chunks = self._merge_chunks_by_image(image_chunks)
            
            # Combine video chunks and image chunks
            chunks.extend(image_chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in ChatGPT chunking: {str(e)}")
            logger.warning("Falling back to simple chunking for images")
            
            # Fallback: Simple chunking - divide remaining story evenly among image prompts
            remaining_sentences = all_sentences[video_prompt_count:]
            sentences_per_image = max(1, len(remaining_sentences) // len(image_prompts))
            
            for i, prompt in enumerate(image_prompts):
                start_idx = i * sentences_per_image
                end_idx = start_idx + sentences_per_image if i < len(image_prompts) - 1 else len(remaining_sentences)
                
                if start_idx < len(remaining_sentences):
                    chunk_text = ' '.join(remaining_sentences[start_idx:end_idx])
                    chunks.append({
                        "chunk": chunk_text,
                        "image": prompt
                    })
            
            return chunks
    
    def _merge_chunks_by_image(self, json_list):
        """Merge all chunks that correspond to the same image together."""
        merged = {}
        
        for item in json_list:
            image = item["image"]
            chunk = item["chunk"]
            
            if image not in merged:
                merged[image] = chunk
            else:
                merged[image] += " " + chunk  # Add a space between merged chunks
        
        # Convert the merged dictionary back to a list of dictionaries
        merged_list = [{"chunk": chunk, "image": image} for image, chunk in merged.items()]
        
        return merged_list

    def create_montage(self, 
                      story_text: str,
                      audio_path: str,
                      video_paths: List[str],
                      image_paths: List[str],
                      prompts: List[str],
                      first_k_sentences: List[str],
                      output_path: str) -> str:
        """Create synchronized video montage with precise timing"""
        try:
            self.clean_temp_files()
            
            # Log the counts of prompts, videos, and images for debugging
            video_prompt_count = len(first_k_sentences)
            image_prompt_count = len(prompts) - video_prompt_count
            total_prompt_count = len(prompts)
            
            logger.info(f"Input counts:")
            logger.info(f"  - Total prompts: {total_prompt_count}")
            logger.info(f"  - Video prompts: {video_prompt_count}")
            logger.info(f"  - Image prompts: {image_prompt_count}")
            logger.info(f"  - Video files: {len(video_paths)}")
            logger.info(f"  - Image files: {len(image_paths)}")
            
            # First validate all input files exist
            missing_videos = [p for p in video_paths if not os.path.exists(p)]
            if missing_videos:
                logger.warning(f"Missing video files: {missing_videos}")
                # Filter out missing videos
                video_paths = [p for p in video_paths if os.path.exists(p)]

            missing_images = [p for p in image_paths if not os.path.exists(p)]
            if missing_images:
                logger.warning(f"Missing image files: {missing_images}")
                # Filter out missing images  
                image_paths = [p for p in image_paths if os.path.exists(p)]

            if not video_paths and not image_paths:
                raise FileNotFoundError("No valid video or image files found")
                
            # Validate that we have enough media files for the prompts
            if len(video_paths) < video_prompt_count:
                logger.warning(f"Not enough video files ({len(video_paths)}) for video prompts ({video_prompt_count})")
                
            if len(image_paths) < image_prompt_count:
                logger.warning(f"Not enough image files ({len(image_paths)}) for image prompts ({image_prompt_count})")
            
            # Get precise audio duration
            audio = AudioSegment.from_mp3(audio_path)
            total_duration = len(audio) / 1000.0  # Convert to seconds
            logger.info(f"Total audio duration: {total_duration:.3f} seconds")
            
            # Analyze text segments with timing information using the first k sentences
            # This will ensure exact matching between prompts and segments
            visual_segments = self.analyze_text_segments(story_text, first_k_sentences, prompts, total_duration)
            
            # Prepare video parts with precise timing
            video_parts = []
            total_video_duration = 0
            
            # Count actual video and image segments
            video_segment_count = sum(1 for segment in visual_segments if segment.get("is_video", False))
            image_segment_count = len(visual_segments) - video_segment_count
            
            logger.info(f"Segment counts after analysis:")
            logger.info(f"  - Total segments: {len(visual_segments)}")
            logger.info(f"  - Video segments: {video_segment_count}")
            logger.info(f"  - Image segments: {image_segment_count}")
            
            # Process segments with precise timing for perfect audio-visual sync
            video_index = 0
            image_index = 0
            
            # Create a list to track segment timing information for final assembly
            segment_timing_info = []
            
            for i, segment in enumerate(visual_segments):
                output_segment = f"{self.temp_dir}/segment_{i}.mp4"
                segment_duration = segment["duration"]
                is_video = segment.get("is_video", False)
                prompt = segment.get("prompt", "")
                
                # Get precise timing information for audio synchronization
                start_time = segment.get("start_time", 0)
                end_time = segment.get("end_time", 0)
                actual_audio_duration = segment.get("actual_audio_duration", segment_duration)
                
                logger.info(f"Processing segment {i + 1}")
                logger.info(f"Target duration: {segment_duration:.3f}s")
                logger.info(f"Audio timing: {start_time:.3f}s to {end_time:.3f}s")
                logger.info(f"Segment type: {'Video' if is_video else 'Image'}")
                logger.info(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Prompt: {prompt}")
                
                if is_video and video_index < len(video_paths):
                    # For video segments, use the video file
                    visual_path = video_paths[video_index]
                    video_index += 1
                    
                    # Trim video to exactly match the audio segment duration for perfect sync
                    self._trim_video(visual_path, output_segment, segment_duration)
                    logger.info(f"Created video segment with duration: {segment_duration:.3f}s")
                else:
                    # For image segments, create a video from an image with the exact segment duration
                    if image_index < len(image_paths):
                        visual_path = image_paths[image_index]
                        image_index += 1
                    else:
                        # If we run out of image paths, reuse the last one
                        if len(image_paths) > 0:
                            logger.warning(f"Ran out of image files, reusing the last one for segment {i+1}")
                            visual_path = image_paths[-1]
                        else:
                            # Fallback if no images available
                            logger.error("No image paths available!")
                            continue
                    
                    # Create video from image with precise duration matching the audio segment
                    self._image_to_video(visual_path, output_segment, segment_duration)
                    logger.info(f"Created video from image with duration: {segment_duration:.3f}s")
                
                # Verify segment duration
                actual_duration = self.get_video_duration(output_segment)
                logger.info(f"Actual segment duration: {actual_duration:.3f}s")
                
                # Store timing information for this segment
                segment_timing_info.append({
                    "segment_path": output_segment,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": actual_duration,
                    "text": segment["text"],
                    "prompt": prompt,
                    "is_video": is_video
                })
                
                total_video_duration += actual_duration
                video_parts.append(output_segment)
            
            # Log total durations before concatenation
            logger.info(f"Total audio duration: {total_duration:.3f}s")
            logger.info(f"Total video duration: {total_video_duration:.3f}s")
            logger.info(f"Video segments used: {video_index}")
            logger.info(f"Image segments used: {image_index}")
            
            # Concatenate with precise timing using segment timing information
            self._concatenate_videos(video_parts, audio_path, output_path, segment_timing_info)
            
            # Verify final output
            final_duration = self.get_video_duration(output_path)
            logger.info(f"Final video duration: {final_duration:.3f}s")
            logger.info(f"Audio-visual synchronization complete with precise timing")
            
            # Display the full story text with segment information
            self._display_story_text_with_segments(segment_timing_info)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error in video montage creation: {str(e)}")
            raise
        finally:
            self.clean_temp_files()

    def _copy_video(self, input_path: str, output_path: str):
        """Copy video maintaining original duration with improved error handling"""
        try:
            # Simple copy command with minimal processing
            copy_cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-c', 'copy',  # Use copy to avoid transcoding
                output_path
            ]
            
            # Run with error capture
            result = subprocess.run(
                copy_cmd, 
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg copy error: {result.stderr}")
                
                # Try with transcoding as fallback
                transcode_cmd = [
                    'ffmpeg', '-y',
                    '-i', input_path,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-pix_fmt', 'yuv420p',
                    output_path
                ]
                
                logger.info("Attempting transcode as fallback")
                result = subprocess.run(transcode_cmd, check=True)
                
        except Exception as e:
            logger.error(f"Error in _copy_video: {str(e)}")
            raise

    def _trim_video(self, input_path: str, output_path: str, target_duration: float):
        """Trim video with precise duration with improved error handling"""
        try:
            # First verify the input file exists and is readable
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input video file does not exist: {input_path}")
            
            if not os.access(input_path, os.R_OK):
                raise PermissionError(f"Input video file is not readable: {input_path}")
            
            # Get video duration to validate trimming is possible
            input_duration = self.get_video_duration(input_path)
            logger.info(f"Input video duration: {input_duration:.3f}s, Target trim duration: {target_duration:.3f}s")
            
            # If video is shorter than target, just copy it
            if input_duration <= target_duration:
                logger.info(f"Video is shorter than target duration, copying instead of trimming")
                self._copy_video(input_path, output_path)
                return
            
            # Use simpler ffmpeg command with explicit error handling
            trim_cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-t', f"{target_duration:.3f}",
                '-c:v', 'libx264',
                '-preset', 'fast',  # Change from medium to fast for better compatibility
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            
            # Run command with full error output capture
            result = subprocess.run(
                trim_cmd, 
                check=False,  # Don't raise exception immediately
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg trim error: {result.stderr}")
                # Fall back to copying if trimming fails
                logger.info("Falling back to video copy due to trim failure")
                self._copy_video(input_path, output_path)
                
        except Exception as e:
            logger.error(f"Error in _trim_video: {str(e)}")
            # Attempt copy as fallback for any error
            try:
                logger.info(f"Attempting to copy video as fallback: {input_path}")
                self._copy_video(input_path, output_path)
            except Exception as copy_err:
                logger.error(f"Fallback copy also failed: {str(copy_err)}")
                raise

    def _display_story_text_with_segments(self, segment_timing_info):
        """Display story text in terminal with highlighting to show which image/video is being displayed during each part"""
        if not segment_timing_info:
            return
            
        print("\n" + "=" * 80)
        print("STORY TEXT WITH AUDIO-VISUAL SYNCHRONIZATION")
        print("=" * 80)
        
        # Sort segments by start time to ensure proper order
        sorted_segments = sorted(segment_timing_info, key=lambda x: x.get("start_time", 0))
        
        # Count video and image segments
        video_segments = [s for s in sorted_segments if s.get("is_video", False)]
        image_segments = [s for s in sorted_segments if not s.get("is_video", False)]
        
        print(f"\nTotal Segments: {len(sorted_segments)} (Video: {len(video_segments)}, Image: {len(image_segments)})")
        print("=" * 80)
        
        # Initialize counters for video and image numbering
        video_count = 1
        image_count = 1
        
        # Display each segment with highlighting and timing information
        for i, segment in enumerate(sorted_segments):
            segment_text = segment.get("text", "")
            segment_prompt = segment.get("prompt", "No prompt available")
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            duration = end_time - start_time
            is_video = segment.get("is_video", False)
            
            # Determine segment type and number
            if is_video:
                segment_type = "Video"
                segment_number = video_count
                video_count += 1
                # Highlight video segments in cyan
                color_code = "\033[1;36m"
                color_text = "\033[36m"
            else:
                segment_type = "Image"
                segment_number = image_count
                image_count += 1
                # Highlight image segments in yellow
                color_code = "\033[1;33m"
                color_text = "\033[33m"
            
            # Create a highlighted display of this segment with ANSI colors
            print(f"\n{color_code}[{segment_type} {segment_number}] {start_time:.2f}s - {end_time:.2f}s (Duration: {duration:.2f}s)\033[0m")
            print(f"{color_code}Prompt: {segment_prompt[:100]}{'...' if len(segment_prompt) > 100 else ''}\033[0m")
            print(color_text + "-" * 80 + "\033[0m")
            print(f"{color_text}Text:\n{segment_text}\033[0m")
            print(color_text + "-" * 80 + "\033[0m")
        
        print("\nFull synchronized story text with segment markers:")
        print("=" * 80)
        
        # Reset counters for full story display
        video_count = 1
        image_count = 1
        
        # Reconstruct the full story text from segments
        full_story = ""
        
        for segment in sorted_segments:
            segment_text = segment.get("text", "")
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            is_video = segment.get("is_video", False)
            
            # Determine segment type and number
            if is_video:
                segment_type = "Video"
                segment_number = video_count
                video_count += 1
                color_code = "\033[1;36m"
                color_text = "\033[36m"
            else:
                segment_type = "Image"
                segment_number = image_count
                image_count += 1
                color_code = "\033[1;33m"
                color_text = "\033[33m"
            
            # Add segment marker and text
            full_story += f"{color_code}[{segment_type} {segment_number} - {start_time:.2f}s]\033[0m {color_text}{segment_text}\033[0m {color_code}[/{start_time:.2f}-{end_time:.2f}s]\033[0m \n\n"
        
        print(full_story)
        print("=" * 80 + "\n")
        
        # Also output a plain text version for terminals that don't support ANSI colors
        print("\nPlain text version (for terminals without color support):")
        print("-" * 80)
        
        # Reset counters for plain text display
        video_count = 1
        image_count = 1
        
        plain_story = ""
        for segment in sorted_segments:
            segment_text = segment.get("text", "")
            segment_prompt = segment.get("prompt", "No prompt available")
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            is_video = segment.get("is_video", False)
            
            # Determine segment type and number
            if is_video:
                segment_type = "Video"
                segment_number = video_count
                video_count += 1
            else:
                segment_type = "Image"
                segment_number = image_count
                image_count += 1
                
            plain_story += f"[{segment_type} {segment_number}]\nPrompt: {segment_prompt[:100]}{'...' if len(segment_prompt) > 100 else ''}\nTiming: {start_time:.2f}s to {end_time:.2f}s\nText: {segment_text}\n\n"
        
        print(plain_story)
        print("-" * 80 + "\n")
    
    def _image_to_video(self, input_path: str, output_path: str, duration: float):
        """Convert image to video with precise duration"""
        try:
            # Verify the input file exists
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input image file does not exist: {input_path}")
                
            # Calculate zoom parameters for smooth motion
            zoom_filter = (
                f"zoompan=z='min(zoom+0.0015,1.05)':"
                f"d={int(duration*60)}:"
                "x='iw/2-(iw/zoom/2)':"
                "y='ih/2-(ih/zoom/2)':"
                "fps=60"
            )
            
            filter_complex = (
                f"{zoom_filter},"
                "scale=1920:1080:force_original_aspect_ratio=decrease,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
            )
            
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', input_path,
                '-t', f"{duration:.3f}",
                '-vf', filter_complex,
                '-c:v', 'libx264',
                '-preset', 'fast',  # Changed from medium to fast
                '-tune', 'stillimage',
                '-pix_fmt', 'yuv420p',
                '-r', '60',
                output_path
            ]
            
            # Run with error capture
            result = subprocess.run(
                cmd, 
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg image-to-video error: {result.stderr}")
                
                # Try simpler command as fallback
                simple_cmd = [
                    'ffmpeg', '-y',
                    '-loop', '1',
                    '-i', input_path,
                    '-t', f"{duration:.3f}",
                    '-vf', "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-pix_fmt', 'yuv420p',
                    output_path
                ]
                
                logger.info("Attempting simpler image conversion as fallback")
                subprocess.run(simple_cmd, check=True)
                
        except Exception as e:
            logger.error(f"Error in _image_to_video: {str(e)}")
            raise

    def _concatenate_videos(self, video_parts: List[str], audio_path: str, output_path: str, segment_timing_info=None):
        """Concatenate videos with precise timing and audio sync using segment timing information"""
        try:
            if not video_parts:
                raise ValueError("No video parts to concatenate")
                
            concat_file = f"{self.temp_dir}/concat.txt"
            temp_video = f"{self.temp_dir}/temp_concat.mp4"
            
            # Verify all parts exist
            missing_parts = [p for p in video_parts if not os.path.exists(p)]
            if missing_parts:
                raise FileNotFoundError(f"Missing video parts: {missing_parts}")
            
            # If we have segment timing info, we'll use it for precise synchronization
            use_precise_sync = segment_timing_info is not None and len(segment_timing_info) > 0
            
            # Display story text with segment highlighting for debugging
            if use_precise_sync and segment_timing_info:
                self._display_story_text_with_segments(segment_timing_info)
            
            if use_precise_sync:
                logger.info("Using precise segment timing for perfect audio-visual synchronization")
                
                # Create a complex filter for precise timing
                filter_complex = []
                inputs = []
                
                # First, extract the audio from the original audio file
                audio_temp = f"{self.temp_dir}/audio_temp.wav"
                extract_audio_cmd = [
                    'ffmpeg', '-y',
                    '-i', audio_path,
                    '-vn', '-acodec', 'pcm_s16le',
                    audio_temp
                ]
                
                subprocess.run(extract_audio_cmd, check=True)
                
                # Process each segment with precise timing
                for i, segment in enumerate(segment_timing_info):
                    segment_path = segment["segment_path"]
                    start_time = segment["start_time"]
                    end_time = segment["end_time"]
                    duration = end_time - start_time
                    
                    # Create a trimmed audio segment
                    segment_audio = f"{self.temp_dir}/segment_audio_{i}.wav"
                    trim_audio_cmd = [
                        'ffmpeg', '-y',
                        '-i', audio_temp,
                        '-ss', f"{start_time:.3f}",
                        '-t', f"{duration:.3f}",
                        segment_audio
                    ]
                    
                    subprocess.run(trim_audio_cmd, check=True)
                    
                    # Create a video with the exact audio segment
                    segment_with_audio = f"{self.temp_dir}/segment_with_audio_{i}.mp4"
                    combine_cmd = [
                        'ffmpeg', '-y',
                        '-i', segment_path,
                        '-i', segment_audio,
                        '-c:v', 'libx264',
                        '-preset', 'fast',
                        '-c:a', 'aac',
                        '-strict', 'experimental',
                        '-map', '0:v:0',
                        '-map', '1:a:0',
                        segment_with_audio
                    ]
                    
                    subprocess.run(combine_cmd, check=True)
                    
                    # Add to our list of segments with synchronized audio
                    video_parts[i] = segment_with_audio
            
            # Write the concat file with our updated segments
            with open(concat_file, 'w', encoding='utf-8') as f:
                for part in video_parts:
                    f.write(f"file '{os.path.abspath(part)}'\n")
            
            # Concatenate videos
            concat_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-pix_fmt', 'yuv420p',
            ]
            
            # If we used precise sync, we already have audio in each segment
            if use_precise_sync:
                concat_cmd.extend([
                    '-c:a', 'aac',
                    output_path
                ])
            else:
                # Standard method - concatenate video then add audio
                concat_cmd.append(temp_video)
            
            # Run with error capture
            result = subprocess.run(
                concat_cmd, 
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg concatenation error: {result.stderr}")
                raise RuntimeError("Video concatenation failed")
            
            # If we didn't use precise sync, add the audio track now
            if not use_precise_sync:
                # Verify temp video was created
                if not os.path.exists(temp_video) or os.path.getsize(temp_video) == 0:
                    raise FileNotFoundError("Concatenated video file was not created properly")
                
                # Add audio with standard sync
                final_cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_video,
                    '-i', audio_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-strict', 'experimental',
                    '-shortest',
                    '-async', '1',
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    output_path
                ]
                
                # Run with error capture
                result = subprocess.run(
                    final_cmd, 
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg audio addition error: {result.stderr}")
                    # Try alternative method if adding audio fails
                    alt_cmd = [
                        'ffmpeg', '-y',
                        '-i', temp_video,
                        '-i', audio_path,
                        '-c:v', 'libx264',  # Re-encode video
                        '-preset', 'fast',
                        '-c:a', 'aac',
                        '-strict', 'experimental',
                        '-shortest',
                        output_path
                    ]
                    
                    logger.info("Attempting alternative audio addition method")
                    subprocess.run(alt_cmd, check=True)
            
            # Clean up temp files
            if os.path.exists(temp_video):
                os.remove(temp_video)
            if os.path.exists(concat_file):
                os.remove(concat_file)
            
            logger.info("Video montage created with perfect audio-visual synchronization")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error during concatenation: {e.stderr if hasattr(e, 'stderr') else str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in video concatenation: {str(e)}")
            raise