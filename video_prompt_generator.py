import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'], model_name="gpt-4", temperature=0.7)

# System message for video prompt generation
system_message = """You are an expert at creating detailed image prompts for text-to-image AI models. Your task is to analyze a story and generate specific, detailed prompts that would work well with text-to-image AI models.

For each prompt:
1. Focus on visual elements and scenes
2. Include specific details about lighting, atmosphere, and mood
3. Specify camera angles and movements where relevant
4. Include character descriptions and emotions
5. Keep each prompt concise but detailed (2-3 sentences)
6. Make prompts specific enough for AI image generation
7. Include relevant visual style references

Format each prompt as a numbered list item. Each prompt should be self-contained and able to generate a specific scene or sequence from the story."""

def calculate_number_of_prompts(story, max_prompts=15):
    """
    Calculate the number of prompts to generate based on story length.
    
    Args:
        story (str): The input story
        max_prompts (int): Maximum number of prompts to generate
    
    Returns:
        int: Number of prompts to generate
    """
    # Count words in the story
    word_count = len(story.split())
    
    # Calculate number of prompts based on story length ranges
    if word_count < 50:  # Extremely short stories
        return 2  # 1 video, 1 image
    elif 50 <= word_count <= 250:  # Very short stories
        return 2  # 1 video, 1 image
    elif 250 < word_count <= 1000:  # Short stories
        return 4  # 2 videos, 1 image
    elif 1000 < word_count <= 2500:  # Medium stories
        return 8  # 3 videos, 5 images
    else:  # Long stories (2500+ words)
        return 15  # 3 videos, 12 images

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
        # Calculate number of prompts dynamically
        num_prompts = calculate_number_of_prompts(story)
        
        # Create the prompt for the LLM
        prompt = f"""
        {system_message}
        
        Story:
        {story}
        
        Generate {num_prompts} detailed images prompts that would work well with text-to-image AI models.
        """
        
        # Generate prompts using the LLM
        response = llm.invoke(prompt)
        prompts = response.content.split('\n')
        
        # Clean up the prompts (remove empty lines and ensure proper formatting)
        prompts = [p.strip() for p in prompts if p.strip()][:num_prompts]
        
        return {
            'prompts': prompts,
            'num_prompts': num_prompts
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
        
        print(f"Number of Prompts Generated: {result['num_prompts']}")
        print("Prompts:")
        for j, prompt in enumerate(result['prompts'], 1):
            print(f"{j}. {prompt}")