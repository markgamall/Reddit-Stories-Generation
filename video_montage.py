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
                logger.info("Using OpenAI Whisper API for transcription with word timestamps")
                with open(audio_path, "rb") as audio_file:
                    response = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                        timestamp_granularities=["word"]
                    )
                
                # Process word timestamps from OpenAI response
                if hasattr(response, 'words') and response.words:
                    for word_data in response.words:
                        word = word_data.word.strip()
                        start = word_data.start
                        end = word_data.end
                        word_timings.append((word, start, end-start))
                        word_to_time[word] = (start, end)
                    logger.info(f"Successfully extracted {len(word_timings)} word timings from OpenAI Whisper API")
                else:
                    logger.warning("OpenAI Whisper API did not return word timestamps")
            except Exception as e:
                logger.warning(f"OpenAI Whisper API error: {e}, falling back to estimation.")
        
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
            prompt_index = chunk.get("prompt_index", -1)  # Get the original prompt index
            
            # Determine if this is a video prompt based on the prompt_index
            is_video = prompt_index < video_segment_count
            
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
            
            # Add segment with the correct prompt_index
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
                "word_count": word_count,
                "prompt_index": prompt_index  # Store the original prompt index
            })
            
            current_position = chunk_start_pos + len(chunk_text)
        
        # Sort segments by start time to ensure proper ordering
        group_segments.sort(key=lambda x: x.get("start_time", 0))
        
        # Log final segment count
        video_segments = sum(1 for segment in group_segments if segment.get("is_video", False))
        image_segments = len(group_segments) - video_segments
        logger.info(f"Created {len(group_segments)} segments (Video: {video_segments}, Image: {image_segments})")
        
        return group_segments
        
    def _chunk_story_with_chatgpt(self, story_text: str, first_k_sentences: List[str], prompts: List[str]) -> List[Dict]:
        """Use ChatGPT to intelligently chunk the story based on relevance to image prompts.
        First k sentences are assigned to videos, and the rest are chunked for images.
        Returns chunks with only the prompts that were successfully matched to text.
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
                    "image": prompt,
                    "prompt_index": i  # Store original prompt index for video prompts
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
        
        1. You will receive a story text along with a LIST OF NUMBERED IMAGE GENERATION PROMPTS.
        
        2. Your goal is to chunk the story based on the following exact rules:
        - Carefully read and match the sentences to the image generation prompts.
        - Whenever you find a sentence that is **very relevant** or **similar** to an image generation prompt:
        - Split a new chunk starting from that sentence.
        - Keep that sentence and all sentences that come after it together in the same chunk, **until** you find another sentence that is highly relevant to the next image generation prompt.
        - When you find another matching sentence, split again into a new chunk.
        - Continue this process until the end of the story.
        
        3. Very important rules:
        - **NEVER miss**, **remove**, **reorder**, or **modify** any part of the story text. All the original text must be present exactly as it is.
        - Never skip or miss any image prompt.
        - Only chunk the story into parts based on the logic above.
        - Each JSON element must contain:
            - A "chunk" field: The exact text of that part of the story.
            - An "image" field: The associated image prompt (copy the full prompt text exactly).
            - An "prompt_index" field: THE ORIGINAL NUMBER OF THE PROMPT (0-indexed) as it appears in the list I provide.
        
        4. Format your final output as a JSON list where each element has three fields: "chunk", "image", and "prompt_index".
        
        Be extremely careful and precise with the prompt_index field - it must match the exact index of the prompt as given in my list.
        """
        
        try:
            # Create the numbered image prompts
            numbered_image_prompts = []
            for idx, prompt in enumerate(image_prompts):
                # Use video_prompt_count as offset to get the original index in the full prompts list
                original_idx = idx + video_prompt_count
                numbered_image_prompts.append(f"Prompt #{original_idx}: {prompt}")

            logger.info("############################################")
            logger.info(json.dumps(numbered_image_prompts))
            logger.info(numbered_image_prompts)
            logger.info("############################################")
            
            # Call ChatGPT to chunk the story
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Story text: {remaining_story}\n\nNumbered image prompts: {json.dumps(numbered_image_prompts)}"}
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0,
                max_tokens=4000
            )
            
            # Extract and parse the response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            # Parse the JSON response
            image_chunks = json.loads(response_text)
            
            # Track which prompts were actually used
            used_prompt_indices = set()
            
            # Validate the prompt_index in each chunk to ensure it's within range
            for chunk in image_chunks:
                if "prompt_index" not in chunk:
                    # If no prompt_index, try to match it based on the image text
                    for idx, prompt in enumerate(image_prompts):
                        if chunk["image"] == prompt:
                            chunk["prompt_index"] = idx + video_prompt_count
                            used_prompt_indices.add(chunk["prompt_index"])
                            break
                    else:
                        # Default to the first image prompt if no match found
                        chunk["prompt_index"] = video_prompt_count
                        used_prompt_indices.add(video_prompt_count)
                else:
                    # Ensure prompt_index is within valid range
                    chunk["prompt_index"] = max(video_prompt_count, min(len(prompts) - 1, chunk["prompt_index"]))
                    used_prompt_indices.add(chunk["prompt_index"])
            
            # Log which prompts were used and which weren't
            all_prompt_indices = set(range(len(prompts)))
            unused_prompt_indices = all_prompt_indices - used_prompt_indices
            if unused_prompt_indices:
                logger.info(f"The following prompts were not matched to any text: {sorted(list(unused_prompt_indices))}")
            
            # Sort chunks by their prompt_index to maintain order
            image_chunks.sort(key=lambda x: x.get("prompt_index", video_prompt_count))
            
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
                        "image": prompt,
                        "prompt_index": i + video_prompt_count  # Store original prompt index
                    })
            
            return chunks
    
    def _merge_chunks_by_image(self, json_list):
        """Merge all chunks that correspond to the same image together."""
        merged = {}
        
        for item in json_list:
            image = item["image"]
            chunk = item["chunk"]
            prompt_index = item.get("prompt_index", 0)  # Get prompt_index, default to 0
            
            if image not in merged:
                merged[image] = {"chunk": chunk, "prompt_index": prompt_index}
            else:
                merged[image]["chunk"] += " " + chunk  # Add a space between merged chunks
        
        # Convert the merged dictionary back to a list of dictionaries
        merged_list = [
            {"chunk": data["chunk"], "image": image, "prompt_index": data["prompt_index"]} 
            for image, data in merged.items()
        ]
        
        # Sort by prompt_index to maintain original order
        merged_list.sort(key=lambda x: x.get("prompt_index", 0))
        
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
            visual_segments = self.analyze_text_segments(story_text, first_k_sentences, prompts, total_duration, audio_path)
            
            # Check which prompts were actually used in the segments
            used_prompt_indices = set(segment.get("prompt_index", -1) for segment in visual_segments)
            
            # Filter out segments with unused prompts
            visual_segments = [segment for segment in visual_segments if segment.get("prompt_index", -1) in used_prompt_indices]
            
            # Count actual video and image segments
            video_segment_count = sum(1 for segment in visual_segments if segment.get("is_video", False))
            image_segment_count = len(visual_segments) - video_segment_count
            
            logger.info(f"Segment counts after analysis:")
            logger.info(f"  - Total segments: {len(visual_segments)}")
            logger.info(f"  - Video segments: {video_segment_count}")
            logger.info(f"  - Image segments: {image_segment_count}")
            logger.info(f"  - Used prompt indices: {sorted(list(used_prompt_indices))}")
            
            # Process segments with precise timing for perfect audio-visual sync
            video_index = 0
            image_index = 0
            
            # Create a list to track segment timing information for final assembly
            segment_timing_info = []
            video_parts = []
            
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
                
                # Get the prompt index for this segment
                prompt_index = segment.get("prompt_index", -1)
                
                if is_video and video_index < len(video_paths):
                    # For video segments, use the video file
                    visual_path = video_paths[video_index]
                    video_index += 1
                    
                    # Trim video to exactly match the audio segment duration for perfect sync
                    self._trim_video(visual_path, output_segment, segment_duration)
                    logger.info(f"Created video segment with duration: {segment_duration:.3f}s for prompt index {prompt_index}")
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
                    logger.info(f"Created video from image with duration: {segment_duration:.3f}s for prompt index {prompt_index}")
                
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
                
                video_parts.append(output_segment)
            
            # Log total durations before concatenation
            logger.info(f"Total audio duration: {total_duration:.3f}s")
            logger.info(f"Video segments used: {video_index}")
            logger.info(f"Image segments used: {image_index}")
            
            # Use the new frame-perfect concatenation method for 100% accurate synchronization
            self._concatenate_videos_frame_perfect(video_parts, audio_path, output_path, segment_timing_info)
            
            # Verify final output
            final_duration = self.get_video_duration(output_path)
            logger.info(f"Final video duration: {final_duration:.3f}s")
            logger.info(f"Frame-perfect audio-visual synchronization complete")
            
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
        for segment in sorted_segments:
            segment_text = segment.get("text", "")
            segment_prompt = segment.get("prompt", "No prompt available")
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            duration = end_time - start_time
            is_video = segment.get("is_video", False)
            prompt_index = segment.get("prompt_index", -1)  # Get original prompt index if available
            
            # Use prompt_index + 1 to get the 1-indexed prompt number
            prompt_number = prompt_index + 1 if prompt_index >= 0 else "unknown"
            
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
            # Use the prompt_number to ensure prompt numbers match the original prompt list
            print(f"\n{color_code}[{segment_type} {segment_number} (Prompt {prompt_number})] {start_time:.2f}s - {end_time:.2f}s (Duration: {duration:.2f}s)\033[0m")
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
            prompt_index = segment.get("prompt_index", -1)
            
            # Get the 1-indexed prompt number
            prompt_number = prompt_index + 1 if prompt_index >= 0 else "unknown"
            
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
            
            # Add segment marker and text with prompt number
            full_story += f"{color_code}[{segment_type} {segment_number} (Prompt {prompt_number}) - {start_time:.2f}s]\033[0m {color_text}{segment_text}\033[0m {color_code}[/{start_time:.2f}-{end_time:.2f}s]\033[0m \n\n"
        
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
            prompt_index = segment.get("prompt_index", -1)
            
            # Get the 1-indexed prompt number
            prompt_number = prompt_index + 1 if prompt_index >= 0 else "unknown"
            
            # Determine segment type and number
            if is_video:
                segment_type = "Video"
                segment_number = video_count
                video_count += 1
            else:
                segment_type = "Image"
                segment_number = image_count
                image_count += 1
                
            plain_story += f"[{segment_type} {segment_number} (Prompt {prompt_number})]\nPrompt: {segment_prompt[:100]}{'...' if len(segment_prompt) > 100 else ''}\nTiming: {start_time:.2f}s to {end_time:.2f}s\nText: {segment_text}\n\n"
        
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

    def _concatenate_videos_frame_perfect(self, video_parts: List[str], audio_path: str, output_path: str, segment_timing_info=None):
        """Create video with precise timing using MoviePy for perfect synchronization"""
        try:
            if not segment_timing_info:
                raise ValueError("Segment timing information is required for precise synchronization")
                
            # Import MoviePy here to avoid adding it as a global dependency
            from moviepy.editor import ImageClip, VideoFileClip, AudioFileClip, CompositeVideoClip
            
            logger.info("=" * 60)
            logger.info("PRECISE TIMING SYNCHRONIZATION WITH MOVIEPY")
            logger.info("=" * 60)
            
            # Sort segments by start time
            sorted_segments = sorted(segment_timing_info, key=lambda x: x.get("start_time", 0))
            total_duration = max(segment["end_time"] for segment in sorted_segments)
            
            logger.info(f"Total segments to position: {len(sorted_segments)}")
            logger.info(f"Total content duration: {total_duration:.3f}s")
            
            # Load the audio file
            audio_clip = AudioFileClip(audio_path)
            audio_duration = audio_clip.duration
            logger.info(f"Audio duration: {audio_duration:.3f}s")
            
            # Create clips list for each segment with exact start and end times
            clips = []
            
            for i, segment in enumerate(sorted_segments):
                start_time = segment["start_time"]
                end_time = segment["end_time"]
                segment_path = segment["segment_path"]
                duration = end_time - start_time
                is_video = segment.get("is_video", False)
                
                logger.info(f"Processing segment {i+1} ({'Video' if is_video else 'Image'}):")
                logger.info(f"  - Start time: {start_time:.3f}s")
                logger.info(f"  - End time: {end_time:.3f}s")
                logger.info(f"  - Duration: {duration:.3f}s")
                
                # Skip invalid segments
                if not os.path.exists(segment_path) or duration <= 0:
                    logger.warning(f"Skipping invalid segment: {segment_path}")
                    continue
                
                # Create clip based on whether it's a video or image
                if is_video:
                    # For video segments, use VideoFileClip
                    clip = VideoFileClip(segment_path).set_duration(duration)
                    logger.info(f"  - Created video clip with duration {duration:.3f}s")
                else:
                    # For image segments, use ImageClip
                    clip = ImageClip(segment_path).set_duration(duration)
                    logger.info(f"  - Created image clip with duration {duration:.3f}s")
                
                # Position clip at exact start time
                clip = clip.set_start(start_time)
                logger.info(f"  - Set clip to start at {start_time:.3f}s")
                
                clips.append(clip)
            
            # Create a background black clip for the full duration
            from moviepy.editor import ColorClip
            background = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=total_duration)
            
            # Add background as the first clip
            clips.insert(0, background)
            
            # Create composite clip with all segments positioned at their exact times
            logger.info(f"Creating composite video with {len(clips)} clips")
            composite_clip = CompositeVideoClip(clips)
            
            # Set the audio
            final_clip = composite_clip.set_audio(audio_clip)
            logger.info(f"Added audio track with duration {audio_duration:.3f}s")
            
            # Write final video
            logger.info(f"Writing output video to {output_path}")
            final_clip.write_videofile(
                output_path, 
                fps=30, 
                codec='libx264', 
                audio_codec='aac',
                preset='fast',
                threads=4
            )
            
            # Clean up MoviePy clips to release resources
            audio_clip.close()
            for clip in clips:
                clip.close()
            composite_clip.close()
            final_clip.close()
            
            # Verify final output
            if os.path.exists(output_path):
                final_duration = self.get_video_duration(output_path)
                logger.info(f"Final video duration: {final_duration:.3f}s (expected: {total_duration:.3f}s)")
                
                if abs(final_duration - audio_duration) > 0.5:
                    logger.warning(f"⚠️ Final video duration differs from audio by {abs(final_duration - audio_duration):.3f}s")
            else:
                logger.error(f"❌ Output video not created: {output_path}")
            
            logger.info("=" * 60)
            logger.info("MoviePy synchronization completed successfully")
            
        except ImportError:
            logger.error("MoviePy is required for this function. Please install it with: pip install moviepy")
            raise
        except Exception as e:
            logger.error(f"Error in MoviePy synchronization: {str(e)}")
            # Fall back to the original method if MoviePy fails
            try:
                logger.warning("Trying fallback concatenation method (FFmpeg)")
                self._concatenate_with_simple_fallback(segment_timing_info, audio_path, output_path)
            except Exception as fallback_e:
                logger.error(f"Fallback concatenation also failed: {str(fallback_e)}")
                raise

    def _concatenate_with_simple_fallback(self, segment_timing_info, audio_path, output_path):
        """Simpler fallback method for creating timed video with fewer overlays at once"""
        try:
            # Sort segments by start time
            sorted_segments = sorted(segment_timing_info, key=lambda x: x.get("start_time", 0))
            
            # We'll create a sequence of video chunks that exactly match our timing needssss
            chunks = []
            current_time = 0
            
            for i, segment in enumerate(sorted_segments):
                start_time = segment["start_time"]
                end_time = segment["end_time"]
                segment_path = segment["segment_path"]
                
                # If there's a gap before this segment, fill with black
                if start_time > current_time:
                    gap_duration = start_time - current_time
                    if gap_duration > 0.1:  # Only fill gaps larger than 100ms
                        logger.info(f"Adding black frame gap: {current_time:.3f}s to {start_time:.3f}s (duration: {gap_duration:.3f}s)")
                        black_chunk = f"{self.temp_dir}/black_{i}.mp4"
                        
                        black_cmd = [
                            'ffmpeg', '-y',
                            '-f', 'lavfi',
                            '-i', f'color=c=black:s=1920x1080:r=60:d={gap_duration}',
                            '-c:v', 'libx264',
                            '-preset', 'ultrafast',
                            '-pix_fmt', 'yuv420p',
                            black_chunk
                        ]
                        subprocess.run(black_cmd, check=True)
                        chunks.append(black_chunk)
                
                # Use exact segment duration
                exact_duration = end_time - start_time
                exact_segment = f"{self.temp_dir}/exact_segment_{i}.mp4"
                
                # Create exact duration segment
                trim_cmd = [
                    'ffmpeg', '-y',
                    '-i', segment_path,
                    '-t', f"{exact_duration:.3f}",
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-pix_fmt', 'yuv420p',
                    exact_segment
                ]
                subprocess.run(trim_cmd, check=True)
                chunks.append(exact_segment)
                
                # Update current time
                current_time = end_time
                
            # Concatenate all chunks
            concat_file = f"{self.temp_dir}/concat_fallback.txt"
            with open(concat_file, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(f"file '{os.path.abspath(chunk)}'\n")
            
            # Concatenate videos
            concat_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-pix_fmt', 'yuv420p',
                f"{self.temp_dir}/video_only.mp4"
            ]
            subprocess.run(concat_cmd, check=True)
            
            # Add audio
            final_cmd = [
                'ffmpeg', '-y',
                '-i', f"{self.temp_dir}/video_only.mp4",
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                output_path
            ]
            subprocess.run(final_cmd, check=True)
            
            logger.info("Fallback concatenation completed successfully")
        
        except Exception as e:
            logger.error(f"Error in fallback concatenation: {str(e)}")
            raise