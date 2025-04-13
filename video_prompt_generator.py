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

# Initialize LLM
llm = ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'], model_name="gpt-4", temperature=0.7)

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

def generate_video_prompts(story, unique_id):
    """
    Generate dynamic number of images prompts from a story using the LLM.
    
    Args:
        story (str): The story to generate prompts from
        unique_id (str): The unique ID of the original story generation
    
    Returns:
        dict: Contains the prompts and metadata
    """
    try:
        # Truncate story to fit within token limits
        truncated_story = truncate_story(story)
        
        # Calculate number of prompts dynamically
        num_video_prompts, num_image_prompts, total_prompts = calculate_number_of_prompts(truncated_story)
        
        # Create the prompt for the LLM with specific instructions
        prompt = f"""
        {system_message}
        
        Story:
        {truncated_story}
        
        Generate exactly {total_prompts} detailed image prompts that accurately represent the most important visual moments in this story.
        
        IMPORTANT INSTRUCTIONS:
        1. The FIRST {num_video_prompts} prompts should represent the FIRST {num_video_prompts} SCENES from the story in chronological order.
           - These will be used for video generation and must show the beginning of the story.
           - Focus on establishing shots, character introductions, and initial plot developments.
        
        2. The REMAINING {num_image_prompts} prompts should represent the KEY MOMENTS from the rest of the story.
           - These will be used for image generation and should capture pivotal scenes.
           - Include major plot points, emotional moments, and the climax.
        
        Ensure all prompts:
        - Maintain visual consistency (characters, style, setting)
        - Are numbered sequentially
        - Directly correspond to specific parts of the story
        """
        
        # Generate prompts using the LLM
        response = llm.invoke(prompt)
        prompts = response.content.split('\n')
        
        # Clean up the prompts (remove empty lines and ensure proper formatting)
        clean_prompts = [p.strip() for p in prompts if p.strip()][:total_prompts]
        
        # Separate video and image prompts
        video_prompts = clean_prompts[:num_video_prompts]
        image_prompts = clean_prompts[num_video_prompts:]
        
        return {
            'video_prompts': video_prompts,
            'image_prompts': image_prompts,
            'num_video_prompts': num_video_prompts,
            'num_image_prompts': num_image_prompts
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