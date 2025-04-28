from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, VideoFileClip
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageAudioVideoCreator:
    def __init__(self):
        self.image_duration = 10  # seconds per image
        self.max_initial_duration = 300  # 5 minutes before using last image
 
    def create_video(self, image_paths, audio_path, output_dir, story_id):
        """
        Create a video from a list of images and an audio file.
        Images rotate every 10 seconds for first 5 minutes, then last image stays until audio ends.
        
        Args:
            image_paths (list): List of paths to image files
            audio_path (str): Path to audio file
            output_dir (str): Directory to save the output video
            story_id (str): ID of the story for unique filename
            
        Returns:
            str: Path to the generated video file
        """
        try:
            logger.info("Starting video creation process")
            
            # Ensure we have at least one image
            if not image_paths or len(image_paths) == 0:
                raise ValueError("No images provided")
            
            # Load audio to get duration
            audio = AudioFileClip(audio_path)
            audio_duration = audio.duration
            
            # Create video clips from images
            image_clips = []
            total_duration = 0
            
            for i, image_path in enumerate(image_paths):
                # For images within first 5 minutes
                if total_duration < self.max_initial_duration:
                    # Calculate duration for this image
                    duration = min(
                        self.image_duration,  # standard duration
                        self.max_initial_duration - total_duration,  # remaining time till 5 min
                        audio_duration - total_duration  # remaining audio time
                    )
                    
                    if duration <= 0:
                        break
                        
                    clip = ImageClip(image_path).set_duration(duration)
                    image_clips.append(clip)
                    total_duration += duration
            
            # If audio is longer than 5 minutes, add final image for remaining duration
            if audio_duration > total_duration:
                final_image = image_paths[-1]  # Use last image
                remaining_duration = audio_duration - total_duration
                final_clip = ImageClip(final_image).set_duration(remaining_duration)
                image_clips.append(final_clip)
            
            # Concatenate all clips
            final_clip = concatenate_videoclips(image_clips)
            
            # Add audio
            final_clip = final_clip.set_audio(audio)
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"final_video_{story_id}_{timestamp}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # Write the final video
            final_clip.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac'
            )
            
            # Clean up
            final_clip.close()
            audio.close()
            for clip in image_clips:
                clip.close()
            
            logger.info(f"Video creation completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating video: {str(e)}")
            raise