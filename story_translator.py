import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'], model_name="gpt-4.1", temperature=0)

# List of supported languages with their codes
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Japanese": "ja",
    "Chinese": "zh",
    "Korean": "ko",
    "Arabic": "ar",
    "Hindi": "hi",
    "Turkish": "tr",
    "Polish": "pl",
    "Vietnamese": "vi",
    "Thai": "th",
    "Greek": "el",
    "Hebrew": "he",
    "Swedish": "sv"
}

# System message for translation
system_message = """You are an expert translator with deep understanding of various languages and cultures. Your task is to translate the given story while preserving:

1. The original meaning and intent
2. Cultural nuances and context
3. Emotional impact and tone
4. Literary devices and style
5. Character voices and dialogue authenticity

Guidelines for translation:
- Maintain the story's narrative flow and pacing
- Preserve idioms and expressions by finding culturally appropriate equivalents
- Keep character personalities consistent
- Retain the original's emotional impact
- Adapt cultural references when necessary for understanding

Your translation should feel natural and fluent in the target language while staying faithful to the original text's essence.

Important considerations:
- If certain concepts don't have direct translations, provide culturally appropriate alternatives
- Maintain any humor, suspense, or emotional elements from the original
- Preserve formatting and paragraph structure
- Keep proper nouns unchanged unless there's a widely accepted translation
"""

def translate_story(story_text, target_language):
    """
    Translate a story into the specified target language.
    
    Args:
        story_text (str): The story to translate
        target_language (str): The target language name (e.g., "Spanish", "French")
        
    Returns:
        str: The translated story
    """
    try:
        logger.info(f"Starting translation to {target_language}")
        
        # Create the prompt for the LLM
        prompt = f"""
        {system_message}
        
        Translate the following story into {target_language}. 
        
        Original Story:
        {story_text}
        
        Provide the translation in {target_language}. Maintain all formatting, spacing, and paragraph breaks.
        """
        
        # Generate translation using the LLM
        response = llm.invoke(prompt)
        translated_text = response.content.strip()
        
        logger.info(f"Successfully translated story to {target_language}")
        
        return translated_text
        
    except Exception as e:
        logger.error(f"Error translating story: {str(e)}")
        raise Exception(f"Translation error: {str(e)}")

def get_supported_languages():
    """
    Get the list of supported languages.
    
    Returns:
        dict: Dictionary of supported languages and their codes
    """
    return SUPPORTED_LANGUAGES