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

# Updated system message for image prompt generation
system_message = system_message = """You are an expert at creating highly detailed image prompts for text-to-image AI models. Your task is to analyze a story and generate vivid, specific prompts that accurately represent key moments from the story.

Each prompt you generate must:
1. Focus on the most visually significant moments that capture the essence of the story.
2. Describe approximately 10 seconds of real-time story progression — imagine each prompt as a short, self-contained scene lasting about 10 seconds.
3. Directly relate to and visually represent a specific part of the story.
4. Ensure absolute consistency across all prompts, including:
    - Character appearance (clothing, facial features, age, hair, body posture, expressions) — describe characters fully in every single prompt.
    - Visual style (e.g., cinematic, painterly, anime — pick one style early and maintain it across all prompts).
    - Color palette and emotional mood (e.g., dark and moody, warm and nostalgic — define early and preserve it).
    - Environmental and world design (architecture, landscape, weather, lighting — must stay coherent throughout).

For each prompt, you must include:
- **Characters**: Describe the main characters with key physical traits, clothing, body language, and emotional expression, **even if the character has already been described before**. Reiterate and maintain full consistency.
- **Setting**: Specify time of day, precise location, and detailed environmental features. Repeat key environmental characteristics to maintain coherence.
- **Action**: Clearly depict what is happening in that specific 10-second slice — focusing on a particular motion, emotion, or interaction.
- **Mood**: Express the emotional tone through atmosphere, lighting, weather, body language, and facial expressions. Keep the emotional consistency aligned with the story's arc.
- **Style**: Define a specific visual style (e.g., cinematic realism, gritty graphic novel, soft watercolor anime) and maintain it firmly across all prompts without switching.

Guidelines for prompt creation:
- Only create prompts for the most crucial, visually-rich scenes that advance the story or deepen emotional impact.
- Each prompt must feel like a continuous "shot" from the same movie or animated film, ensuring full continuity and cohesion.
- Dialogue scenes: Focus visually on facial expressions, gestures, postures, and the emotional tension between characters.
- Emotional scenes: Emphasize body language, close-up framing, lighting, and atmosphere.
- Action scenes: Freeze a dynamic moment of movement, decision, or impact with intense detail.

Prompt Format: Each prompt must:
1. Be numbered sequentially and self-contained, describing a **single 10-second visual scene**.
2. Be 2–4 richly detailed sentences long.
3. Begin with a clear visual framing (e.g., "A close-up of...", "A wide shot showing...", "An aerial view of...").

Concrete rules:
- Integrate full character and environment descriptions into every prompt, even if repetitive, to ensure prompts can stand alone.
- Follow the established tone, style, and environmental logic with absolute discipline.
- Never leave placeholder prompts like "Prompt 2:" or "Image 3:" — if a prompt cannot be written, **skip numbering** without gaps.
- Every prompt must be vivid, rich, and fully descriptive, never vague or empty.

REMEMBER: Each prompt should feel like a short cinematic cut lasting about 10 seconds, fully immersing the viewer into the consistent world you've created.
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
    Calculate the number of image prompts to generate based on story length.

    Args:
        story (str): The input story

    Returns:
        int: Number of image prompts to generate
    """
    # Count words in the story
    word_count = len(story.split())

    if word_count < 100:
        return 5
    if 100 <= word_count <= 350:
        return 10
    if 351 <= word_count <= 500:
        return 15
    if 501 <= word_count < 700:
        return 20
    if word_count >= 700:
        return 30

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

def generate_image_prompts(story, unique_id):
    """
    Generate image prompts that represent key moments from a story.
    
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
        total_prompts = calculate_number_of_prompts(truncated_story)
        
        # Log the prompt count for debugging
        logger.info(f"Generating prompts: {total_prompts} total image prompts")
        
        # Create the prompt for the LLM with specific instructions
        prompt = f"""
        {system_message}
        
        Story:
        {truncated_story}
        
        Generate EXACTLY {total_prompts} detailed image prompts that accurately represent specific visual moments in this story.
        
        IMPORTANT INSTRUCTIONS:
        - Each prompt should represent a key moment from the story
        - These should evenly cover the content of the story
        - Include major plot points and emotional moments
        - You MUST generate EXACTLY {total_prompts} image prompts, no more and no less
        
        Ensure all prompts:
        - Maintain visual consistency (characters, style, setting)
        - Are numbered sequentially from 1 to {total_prompts}
        - Focus on creating a cohesive visual narrative
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
        
        # Log the final prompt count
        logger.info(f"Final prompts: {len(clean_prompts)} total image prompts")
        
        return {
            'image_prompts': clean_prompts,
            'num_image_prompts': len(clean_prompts)
        }
        
    except Exception as e:
        raise Exception(f"Error generating image prompts: {str(e)}")

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
        result = generate_image_prompts(test_story, test_id)
        
        print(f"Image Prompts ({result['num_image_prompts']}):") 
        for j, prompt in enumerate(result['image_prompts'], 1):
            print(f"{j}. {prompt}")