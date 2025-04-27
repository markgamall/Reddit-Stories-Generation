import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import tiktoken
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize LLM with temperature 0 for more consistent outputs
llm = ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'], model_name="gpt-4.1", temperature=0)

# Updated system message for video prompt generation
system_message = """You are an expert at creating detailed image prompts for text-to-image AI models. Your task is to analyze a story and generate specific, detailed prompts that accurately represent key moments from the story.

Each prompt you generate must:
1. Focus on the most visually significant moments that capture the essence of the story
2. Directly relate to and visually represent a specific part of the story
3. Maintain absolute consistency across all prompts, including:
    - Character appearance (clothing, facial features, age, hair, expressions)
    - Visual style (e.g., cinematic, painterly, anime — use the same across all prompts)
    - Color palette and mood (e.g., dark and moody, warm and nostalgic — define early and keep it consistent)
    - Environment and world design (architecture, weather, lighting — all should feel part of one coherent world)

For each prompt, include these essential elements:
- Characters: Describe main characters with key physical traits, clothing, and emotional expression. Be consistent in describing them in every prompt.
- Setting: Include time of day, location, and environmental details. Use the same environmental logic throughout the story.
- Action: Clearly describe what is happening in the scene, focusing on a specific moment in motion or interaction.
- Mood: Convey the emotional tone through lighting, body language, and atmosphere. Keep the emotional palette coherent across the story's arc.
- Style: Define and use a consistent visual style (e.g., cinematic realism, soft painterly, gritty graphic novel, 90s anime). Do not switch styles between prompts.

Guidelines for prompt creation:
- Only create prompts for the most important scenes that visually tell the story
- Maintain full continuity and consistency — as if all scenes are frames from the same movie or animated short
- For dialogue-heavy scenes, focus on facial expressions, postures, and eye contact to show emotional weight
- For emotional scenes, emphasize body language, lighting, and framing
- For action scenes, freeze the moment of impact or decision with dynamic detail

Prompt Format: Each prompt should:
1. Be numbered and self-contained, describing a single scene
2. Be 2–3 sentences long
3. Begin with a clear visual framing (e.g., "A close-up of...", "A wide shot showing...")

Include concrete visual elements from the story and follow the established tone, style, and world logic

IMPORTANT: Never output empty or placeholder prompts such as "Prompt 2:" or "Image 3:". Every prompt must be a detailed, descriptive visual scene. If you cannot generate a prompt for a given number, skip it and do not include an empty label. Do not output any prompt that is just a number or label without a description.
"""

def estimate_tokens(text):
    """
    Estimate the number of tokens in a text using tiktoken.
    
    Args:
        text (str): The text to estimate tokens for
        
    Returns:
        int: Estimated number of tokens
    """
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

def truncate_story(story, max_tokens=6000):
    """
    Truncate a story to fit within token limits while preserving complete sentences.
    
    Args:
        story (str): The story to truncate
        max_tokens (int): Maximum number of tokens allowed
        
    Returns:
        str: Truncated story
    """
    try:
        logger.info(f"Starting story truncation with max_tokens: {max_tokens}")
        
        # Get the encoding once
        encoding = tiktoken.encoding_for_model("gpt-4")
        logger.info("Successfully initialized tiktoken encoding")
        
        # Calculate initial token count
        initial_tokens = len(encoding.encode(story))
        logger.info(f"Initial story token count: {initial_tokens}")
        
        # If story is already within limits, return it
        if initial_tokens <= max_tokens:
            logger.info("Story is within token limits, no truncation needed")
            return story
            
        # Split into sentences (basic implementation)
        sentences = story.replace('\n', ' ').split('. ')
        logger.info(f"Split story into {len(sentences)} sentences")
        
        truncated_story = ''
        current_tokens = 0
        sentences_processed = 0
        
        for sentence in sentences:
            try:
                # Add the sentence and a period
                test_story = truncated_story + sentence + '. '
                # Calculate tokens for the new sentence only
                new_tokens = len(encoding.encode(sentence + '. '))
                
                logger.debug(f"Processing sentence {sentences_processed + 1}:")
                logger.debug(f"- Sentence length: {len(sentence)} chars")
                logger.debug(f"- New tokens: {new_tokens}")
                logger.debug(f"- Current total tokens: {current_tokens}")
                
                if current_tokens + new_tokens > max_tokens:
                    logger.info(f"Reached token limit after {sentences_processed} sentences")
                    logger.info(f"Final token count: {current_tokens}")
                    break
                    
                truncated_story = test_story
                current_tokens += new_tokens
                sentences_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing sentence {sentences_processed + 1}: {str(e)}")
                logger.error(f"Sentence content: {sentence[:100]}...")  # Log first 100 chars of problematic sentence
                raise
        
        logger.info(f"Truncation complete. Processed {sentences_processed} sentences")
        logger.info(f"Final truncated story length: {len(truncated_story)} chars")
        logger.info(f"Final token count: {current_tokens}")
        
        return truncated_story.strip()
        
    except Exception as e:
        logger.error(f"Error in truncate_story: {str(e)}")
        logger.error(f"Story length: {len(story)} chars")
        raise

def calculate_number_of_prompts(story):
    """
    Calculate the number of prompts to generate based on story length.
    
    Args:
        story (str): The input story
    
    Returns:
        tuple: (num_video_prompts, num_image_prompts, total_prompts)
    """
    # Count words in the story
    word_count = len(story.split())
    
    # Calculate number of prompts based on story length ranges
    if word_count < 50:  # Extremely short stories
        return (1, 4, 5)
    elif 50 <= word_count <= 250:  # Very short stories
        return (1, 6, 7)
    elif 250 < word_count <= 1000:  # Short stories
        return (2, 8, 10)
    elif 1000 < word_count <= 2500:  # Medium stories
        return (3, 12, 15)
    else:  # Long stories (2500+ words)
        return (3, 18, 21)

def extract_valid_prompts(raw_prompts, expected_count=None):
    """
    Extracts and cleans prompts, filtering out placeholders and incomplete prompts.
    Ensures the exact number of prompts are returned as expected.
    
    Args:
        raw_prompts (list): List of raw prompt strings (from LLM response)
        expected_count (int, optional): Number of prompts expected (for stricter filtering)
    Returns:
        list: List of valid, detailed prompts matching the expected count
    """
    import re
    valid_prompts = []
    current_prompt = ""
    
    # First pass: extract all numbered prompts
    for line in raw_prompts:
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts with a number followed by period
        if re.match(r'^\d+\.\s', line):
            # If we have a previous prompt, add it to valid prompts
            if current_prompt and len(current_prompt.split()) >= 5:  # Lower threshold for initial extraction
                valid_prompts.append(current_prompt)
            # Start new prompt, removing the number
            current_prompt = re.sub(r'^\d+\.\s*', '', line)
        else:
            # Append line to current prompt if it's not a reference or instruction
            if not any(line.lower().startswith(x) for x in ['for reference', 'ensure all', 'important', 'note:', 'remember:']):
                current_prompt = current_prompt + " " + line if current_prompt else line
    
    # Add the last prompt if it exists
    if current_prompt and len(current_prompt.split()) >= 5:
        valid_prompts.append(current_prompt)
    
    # Filter and clean prompts
    cleaned_prompts = []
    for prompt in valid_prompts:
        cleaned = prompt.strip()
        # Remove any remaining numbering or labels
        cleaned = re.sub(r'^(prompt\s*\d+:?\s*|image\s*\d+:?\s*)', '', cleaned, flags=re.IGNORECASE)
        # Ensure prompt is detailed enough
        if len(cleaned.split()) >= 10 and len(cleaned) >= 50:
            cleaned_prompts.append(cleaned)
        elif len(cleaned) >= 30:  # Accept slightly shorter prompts if we need more
            cleaned_prompts.append(cleaned)
    
    # Log the extraction results
    logger.info(f"Extracted {len(cleaned_prompts)} valid prompts from {len(raw_prompts)} lines of text")
    
    # Handle the case where we don't have enough prompts
    if expected_count is not None:
        if len(cleaned_prompts) < expected_count:
            logger.warning(f"Expected {expected_count} prompts but got {len(cleaned_prompts)} valid prompts.")
            
            # Try to recover more prompts with less strict filtering
            additional_prompts = []
            for prompt in valid_prompts:
                cleaned = prompt.strip()
                cleaned = re.sub(r'^(prompt\s*\d+:?\s*|image\s*\d+:?\s*)', '', cleaned, flags=re.IGNORECASE)
                # Include prompts we previously filtered out for being too short
                if cleaned not in cleaned_prompts and len(cleaned) >= 20:
                    additional_prompts.append(cleaned)
            
            # Add additional prompts until we reach the expected count
            needed = expected_count - len(cleaned_prompts)
            if additional_prompts and needed > 0:
                logger.info(f"Adding {min(needed, len(additional_prompts))} additional prompts with relaxed criteria")
                cleaned_prompts.extend(additional_prompts[:needed])
        
        # If we have too many prompts, trim to the expected count
        elif len(cleaned_prompts) > expected_count:
            logger.warning(f"Got {len(cleaned_prompts)} prompts but only expected {expected_count}. Trimming excess.")
            cleaned_prompts = cleaned_prompts[:expected_count]
    
    # Final check on prompt count
    if expected_count is not None:
        logger.info(f"Final prompt count: {len(cleaned_prompts)}/{expected_count}")
    
    return cleaned_prompts

def generate_video_prompts(story, unique_id):
    """
    Generate video and image prompts with videos for first k sentences.
    Ensures exact matching between the number of prompts and media files.
    
    Args:
        story (str): The story to generate prompts from
        unique_id (str): The unique ID of the original story generation
    
    Returns:
        dict: Contains the prompts, metadata, and first k sentences
    """
    try:
        # Truncate story to fit within token limits
        truncated_story = truncate_story(story)
        
        # Get first k sentences for video prompts
        sentences = truncated_story.replace('\n', ' ').split('. ')
        sentences = [s + '.' for s in sentences if s.strip()]
        
        # Calculate number of prompts dynamically
        num_video_prompts, num_image_prompts, total_prompts = calculate_number_of_prompts(truncated_story)
        
        # Get first k sentences that will be paired with videos
        first_k_sentences = sentences[:num_video_prompts]
        
        # Log the prompt counts for debugging
        logger.info(f"Generating prompts: {total_prompts} total ({num_video_prompts} video, {num_image_prompts} image)")
        
        # Create the prompt for the LLM with specific instructions
        prompt = f"""
        {system_message}
        
        Story:
        {truncated_story}
        
        Generate EXACTLY {total_prompts} detailed image prompts that accurately represent specific visual moments in this story.
        
        IMPORTANT INSTRUCTIONS:
        1. The FIRST {num_video_prompts} prompts MUST each represent EXACTLY ONE of the first {num_video_prompts} SENTENCES from the story:
           - Prompt 1 must represent sentence 1
           - Prompt 2 must represent sentence 2
           - And so on...
           - These will be used for 5-second video clips, so each prompt must represent exactly one sentence.
        
        2. The REMAINING {num_image_prompts} prompts should represent the KEY MOMENTS from the rest of the story.
           - These should evenly cover the remaining content.
           - Include major plot points and emotional moments.
           - You MUST generate EXACTLY {num_image_prompts} image prompts, no more and no less.
        
        For reference, here are the first {num_video_prompts} sentences that need their own video prompts:
        {' '.join(first_k_sentences)}
        
        Ensure all prompts:
        - Maintain visual consistency (characters, style, setting)
        - Are numbered sequentially from 1 to {total_prompts}
        - For video prompts, focus on action and movement that can be shown in 5 seconds
        """
        
        # Generate prompts using the LLM
        response = llm.invoke(prompt)
        prompts = response.content.strip().split('\n')
        
        # Extract and validate prompts
        clean_prompts = extract_valid_prompts(prompts, total_prompts)
        
        # If we didn't get enough prompts, try one more time with a more explicit instruction
        if len(clean_prompts) < total_prompts:
            logger.warning(f"First attempt generated {len(clean_prompts)}/{total_prompts} valid prompts. Trying again...")
            # Add more explicit formatting instructions
            retry_prompt = prompt + f"\n\nVERY IMPORTANT: You MUST generate EXACTLY {total_prompts} prompts. Format each prompt as a numbered item (1., 2., etc.) followed by a detailed visual description. Each prompt must be at least 2 sentences long and include all required elements (characters, setting, action, mood, style)."
            
            response = llm.invoke(retry_prompt)
            prompts = response.content.strip().split('\n')
            clean_prompts = extract_valid_prompts(prompts, total_prompts)
            
            # If still not enough, try one more time with even more explicit instructions
            if len(clean_prompts) < total_prompts:
                logger.warning(f"Second attempt generated {len(clean_prompts)}/{total_prompts} valid prompts. Final attempt...")
                final_prompt = retry_prompt + f"\n\nCRITICAL: You MUST output EXACTLY {total_prompts} numbered prompts (1-{total_prompts}). Each prompt MUST be detailed and descriptive, at least 50 words long."
                
                response = llm.invoke(final_prompt)
                prompts = response.content.strip().split('\n')
                clean_prompts = extract_valid_prompts(prompts, total_prompts)
        
        # Ensure we have exactly the right number of prompts by padding or trimming if necessary
        if len(clean_prompts) < total_prompts:
            logger.warning(f"Could only generate {len(clean_prompts)}/{total_prompts} valid prompts. Padding with placeholders.")
            # Pad with placeholder prompts if we don't have enough
            while len(clean_prompts) < total_prompts:
                placeholder = f"A detailed visual scene from the story showing a key moment with consistent style and character appearance."
                clean_prompts.append(placeholder)
        elif len(clean_prompts) > total_prompts:
            logger.warning(f"Generated {len(clean_prompts)} prompts but only needed {total_prompts}. Trimming excess.")
            # Trim excess prompts
            clean_prompts = clean_prompts[:total_prompts]
        
        # Separate video and image prompts
        video_prompts = clean_prompts[:num_video_prompts]
        image_prompts = clean_prompts[num_video_prompts:total_prompts]
        
        # Log the final prompt counts
        logger.info(f"Final prompts: {len(clean_prompts)} total ({len(video_prompts)} video, {len(image_prompts)} image)")
        
        return {
            'video_prompts': video_prompts,
            'image_prompts': image_prompts,
            'num_video_prompts': len(video_prompts),
            'num_image_prompts': len(image_prompts),
            'first_k_sentences': first_k_sentences
        }
        
    except Exception as e:
        raise Exception(f"Error generating video prompts: {str(e)}")

if __name__ == "__main__":
    # Example usage with different story lengths
    test_stories = [
        # Very short story
        "A cat sat on a mat.",
        
        # Short story
        "In a small town nestled between mountains, a young detective uncovers a mysterious disappearance that challenges everything she knows about her quiet community.",
        
        # Medium story
        """In the heart of a bustling city, Sarah, a talented graphic designer, finds herself trapped in a monotonous corporate job. Her creativity is suffocated by endless meetings and uninspired briefs. One day, a chance encounter with a street artist reignites her passion, pushing her to reclaim her artistic dreams and challenge the status quo.""",
        
        # Long story
        """Imagine this: a demolition expert, pushed to the edge, his car towed one too many times. He snaps. Boom! Buildings crumble. Then, a bride, all dolled up, discovers her hubby's dirty secret. Betrayal on her wedding day? Hell hath no fury. And a waitress, serving the man who ruined her family. Revenge is a dish best served cold, right?

This ain't your average flick, folks. It's 'Wild Tales', an anthology of six stories, each darker, funnier, and more thrilling than the last. It's got Oscar buzz, and for good reason.

But here's the twist: what if these aren't just stories? What if they're warnings? What if we're all just one bad day away from becoming a 'Wild Tale' ourselves?"""
    ]
    
    for i, test_story in enumerate(test_stories, 1):
        print(f"\n--- Test Story {i} ---")
        print(f"Word Count: {len(test_story.split())}")
        
        test_id = f"test{i}"
        result = generate_video_prompts(test_story, test_id)
        
        print(f"Video Prompts ({result['num_video_prompts']}):")
        for j, prompt in enumerate(result['video_prompts'], 1):
            print(f"{j}. {prompt}")
        
        print(f"\nImage Prompts ({result['num_image_prompts']}):")
        for j, prompt in enumerate(result['image_prompts'], result['num_video_prompts'] + 1):
            print(f"{j}. {prompt}")