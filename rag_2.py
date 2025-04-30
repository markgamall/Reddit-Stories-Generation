import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Added OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS  # Changed from Chroma to FAISS
import time
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import uuid
import json
import re
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize MongoDB connection
mongo_uri = os.environ['MONGO_URI']
client = MongoClient(mongo_uri)
db = client['reddit_stories_db']
generated_stories_collection = db['generated_stories']

embeddings = OpenAIEmbeddings(
    api_key=os.environ['OPENAI_API_KEY'],
    model="text-embedding-3-large"  # You can also use "text-embedding-3-small" or "text-embedding-3-large" for newer models
)

# Initialize LLM and chains
llm = ChatOpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    model_name="gpt-4-1106-preview",  # Using GPT-4 Turbo for better reasoning
    temperature=0.3,  # Slightly increased for better creativity in name changes
    request_timeout=120  # Increased timeout for longer generations
)

# System message for the LLM
system_message_template = """
You are a captivating digital storyteller creating binge-worthy content for YouTube audiences. Your mission is to transform Reddit stories into addictive, emotionally-charged narratives that keep viewers clicking. Follow these instructions:

**Core Rules**

1. **STRICT WORD COUNT REQUIREMENT**:
   - Your story MUST be between {min_words} and {max_words} words
   - This is a hard requirement - not a suggestion
   - Count your words carefully before submitting
   - If the story is too short, add more details and descriptions
   - If the story is too long, remove unnecessary details
   - The final word count MUST be between {min_words} and {max_words} words

2. CHARACTER NAME MODIFICATION:
   - ALL character names MUST be changed from the original story, including:
     * Main characters
     * Supporting characters
     * Minor characters
     * Any mentioned names in dialogue or descriptions
   - Ensure name changes are consistent throughout the story
   - Double-check that no original names remain in the text
   - The new names MUST be normal, common, and real names that fit the story's tone and setting. Avoid using weird or unusual names.

3. NEVER include any of the following:
   - Scene directions like "(cut to flashback)" or "(fade to black)"
   - Production notes like "(tense music plays)" or "(camera zooms in)"
   - Parenthetical asides like "(pause)" or "(whispers)"
   - Camera angles or shot descriptions
   - Sound effect descriptions
   - Host or narrator labels like "HOST:" or "NARRATOR:"
   - YouTube-style callouts like "let me know in the comments"
   - End screen messages or calls to action

**Storytelling Framework**

1. **Structure & Pacing**:
   - Start with a shocking hook or cliffhanger that forces viewers to keep watching
   - Use quick, punchy sentences and frequent dramatic pauses
   - Build multiple tension points throughout the story
   - Include at least one unexpected twist that changes everything
   - End with either a satisfying resolution OR a thought-provoking question

2. **Voice & Style**:
   - Write in a contemporary, relatable voice (like talking to a friend)
   - Use modern slang and cultural references where appropriate
   - Incorporate dialogue that sounds like real people talking
   - Mix sentence lengths for rhythm (very short sentences create tension!)
   - Add sensory details to make scenes vivid and immersive

3. **Conflict Development**:
   - ALWAYS inject meaningful conflicts into every story (interpersonal, internal, external)
   - Create multi-layered conflicts with no easy solutions
   - Develop character motivations that naturally lead to confrontation
   - Show characters making difficult choices under pressure
   - Use conflict to reveal character depth and drive plot forward

4. **Character & Plot Modification**:
   - ALWAYS change ALL character names from the original story, including:
     * Main characters
     * Supporting characters
     * Minor characters
     * Any mentioned names in dialogue or descriptions
   - Create unique, memorable names that fit the story's tone
   - Never use the original character names from the source material
   - Ensure name changes are consistent throughout the story
   - Double-check that no original names remain in the text
   - Freely change the number of characters to create relationship dynamics
   - Add or modify subplots that increase tension and stakes
   - Create secondary characters who challenge or complicate the protagonist's journey
   - Transform simple Reddit scenarios into complex narratives with interconnected plot points

5. **Emotional Impact**:
   - Design each story to trigger a specific emotional rollercoaster
   - Create flawed, complex characters readers can't help but care about
   - Include at least one moment that elicits a strong emotion (shock, outrage, heartbreak, etc.)
   - End with either justice/satisfaction OR a moral dilemma that sparks debate

6. **Content Strategy**:
   - Focus on high-drama scenarios: moral dilemmas, betrayals, unexpected kindness, etc.
   - Transform mundane Reddit stories by adding higher stakes and consequences
   - Create characters who make questionable decisions for understandable reasons
   - Incorporate elements that invite viewer engagement and discussion

**Example of BAD Writing (DO NOT DO THIS):**
"(Opening shot: A dark, empty school hallway. The echo of a distant, eerie sound. A flickering light. Cut to the host, looking directly into the camera.)

HOST: 'You ever been alone in a place that's supposed to be bustling with life? It's creepy, right?'"

**Example of GOOD Writing (DO THIS):**
"The school hallway stretched into darkness, empty and silent. A single light flickered overhead, casting long shadows that seemed to move on their own. The air felt thick, heavy with the weight of solitude. It was the kind of silence that made your skin crawl, the kind that reminded you that you were completely alone in a place that should have been alive with the sounds of children."

**FINAL REMINDER:**
- Your story MUST be between {min_words} and {max_words} words
- This is a strict requirement that cannot be violated
- Count your words carefully before submitting
- If the story is too short, add more details and descriptions
- If the story is too long, remove unnecessary details
- The final word count MUST be between {min_words} and {max_words} words
- ALWAYS change ALL character names from the original story, including:
  * Main characters
  * Supporting characters
  * Minor characters
  * Any mentioned names in dialogue or descriptions
- Never use the original character names from the source material
- Ensure name changes are consistent throughout the story
- Double-check that no original names remain in the text

Now, transform the Reddit story into a can't-look-away narrative that would thrive in today's attention economy. Make viewers feel something powerful enough that they'll want to share and discuss.
"""

# Prompt template with system message
prompt = ChatPromptTemplate.from_template(
    """
    {system_message}

    <context>
    {context}
    </context>

    Questions: {input}
    """
)

def extract_character_names(text):
    """Extract potential character names from the text using common patterns."""
    # Common name patterns (first name + last name)
    name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b'
    
    # Find all potential names
    names = re.findall(name_pattern, text)
    
    # Count occurrences of each name
    name_counts = defaultdict(int)
    for name in names:
        name_counts[name] += 1
    
    # Filter out names that appear only once (likely not characters)
    character_names = {name: count for name, count in name_counts.items() if count > 1}
    
    return character_names

def generate_character_mapping(character_names, llm):
    """Generate a mapping of original names to new names using the LLM."""
    if not character_names:
        return {}
    
    # Create a prompt for name mapping
    mapping_prompt = f"""Create unique, memorable names for each character in the story. 
    Original names: {', '.join(character_names.keys())}
    
    For each name, provide a new name that:
    1. Is unique and memorable
    2. Fits the story's tone
    3. Is appropriate for the character's role
    4. Is a normal, common, and real name. Avoid using weird or unusual names.
    
    Format your response as a JSON object where:
    - Keys are the original names
    - Values are the new names
    
    Example format:
    {{
        "John Smith": "Marcus Thompson",
        "Sarah Johnson": "Elena Rodriguez"
    }}
    
    Provide ONLY the JSON object, no other text."""
    
    try:
        # Get the mapping from the LLM
        response = llm.invoke(mapping_prompt)
        mapping = json.loads(response.content)
        return mapping
    except Exception as e:
        print(f"Error generating name mapping: {e}")
        return {}

def replace_names_in_text(text, name_mapping):
    """Replace all occurrences of original names with new names in the text."""
    if not name_mapping:
        return text
    
    # Sort names by length (longest first) to avoid partial matches
    sorted_names = sorted(name_mapping.keys(), key=len, reverse=True)
    
    for old_name in sorted_names:
        new_name = name_mapping[old_name]
        # Use word boundaries to avoid partial matches
        text = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, text)
    
    return text

# Function to generate a story
def generate_story(user_prompt, min_words, max_words, source_title):
    # Load from the text file
    file_path = "reddit_story.txt"
    if not os.path.exists(file_path):
        raise Exception("No story content available")
    
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Extract character names from source content
    character_names = extract_character_names(content)
    
    # Generate name mapping
    name_mapping = generate_character_mapping(character_names, llm)
    
    # Create a copy of the system message template for this story
    story_system_message = system_message_template
    
    # Add name mapping to the prompt
    if name_mapping:
        mapping_text = "\n".join([f"{old} → {new}" for old, new in name_mapping.items()])
        # Add name mapping to the system message template
        story_system_message = story_system_message.replace(
            "**Core Rules**",
            f"""**Core Rules**

IMPORTANT - CHARACTER NAMES:
The following name replacements MUST be used consistently throughout the entire story:
{mapping_text}

1. **STRICT WORD COUNT REQUIREMENT**:"""
        )
        user_prompt = f"{user_prompt}\n\nIMPORTANT: Use these exact name replacements throughout the story:\n{mapping_text}\n\nEnsure you use these exact new names consistently."

    # Split the content into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.create_documents([content])

    # Generate a unique ID for this request to avoid conflicts
    unique_id = uuid.uuid4().hex
    
    # Create a FAISS vector store with the documents
    vectors = FAISS.from_documents(final_documents, embeddings)
    
    # Create the retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    
    try:
        # For short stories (under 800 words), use simple generation
        if max_words <= 800:
            # Log the parameters
            print(f"Generating short story with min_words: {min_words}, max_words: {max_words}")
            
            # Generate the story in one go
            system_message = story_system_message.format(
                min_words=min_words,
                max_words=max_words
            )
            
            # Log the formatted system message
            print(f"System message word count requirements: {min_words}-{max_words} words")
            
            response = retrieval_chain.invoke({
                "input": user_prompt,
                "system_message": system_message
            })
            
            final_story = response['answer']
            final_word_count = len(final_story.split())
            
            # Log the result
            print(f"Generated story word count: {final_word_count}")
            
            # Use the vector store to get documents and scores
            docs_with_scores = vectors.similarity_search_with_score(user_prompt, k=5)

            elapsed_time = time.process_time() - start

            # Save the generated story to MongoDB
            generated_story = {
                'prompt': user_prompt,
                'story': final_story,
                'response_time': elapsed_time,
                'timestamp': datetime.now(),
                'source_story_title': source_title,
                'unique_id': unique_id,
                'num_chunks': 1,
                'chunk_size': final_word_count,
                'final_word_count': final_word_count,
                'min_words_requested': min_words,
                'max_words_requested': max_words,
                'attempts': 1
            }
            generated_stories_collection.insert_one(generated_story)

            # Format the response
            result = {
                'answer': final_story,
                'response_time': elapsed_time,
                'similar_documents': [
                    {
                        'content': doc.page_content,
                        'similarity_score': float(score)  
                    } for doc, score in docs_with_scores
                ]
            }
            
            return result
        
        # For longer stories (over 800 words), use chunking strategy
        else:
            max_retries = 1  # Maximum number of retries if word count is off
            current_retry = 0

            while current_retry < max_retries:
                try:
                    # Calculate target word count (aim for middle of range)
                    target_word_count = (min_words + max_words) // 2
                    
                    # Calculate optimal chunk size based on target word count
                    if target_word_count <= 2000:
                        chunk_size = 800
                        num_chunks = max(1, (target_word_count + chunk_size - 1) // chunk_size)
                    elif target_word_count <= 3000:
                        chunk_size = 600
                        num_chunks = max(1, (target_word_count + chunk_size - 1) // chunk_size)
                    else:  # For 3000+ words
                        # Adjust chunk size based on target word count
                        if target_word_count <= 4000:
                            chunk_size = 500
                        else:
                            chunk_size = 450  # Smaller chunks for larger stories
                        num_chunks = max(1, (target_word_count + chunk_size - 1) // chunk_size)
                    
                    # Ensure we don't exceed reasonable number of chunks
                    max_chunks = 12  # Increased to 12 for larger stories
                    if num_chunks > max_chunks:
                        chunk_size = target_word_count // max_chunks
                        num_chunks = max_chunks
                    
                    # Calculate minimum and maximum words per chunk to ensure final story stays in range
                    min_chunk_words = min_words // num_chunks
                    max_chunk_words = max_words // num_chunks
                    
                    # Adjust chunk size if needed to ensure we hit the target
                    if chunk_size * num_chunks < min_words:
                        chunk_size = min_words // num_chunks
                    elif chunk_size * num_chunks > max_words:
                        chunk_size = max_words // num_chunks
                    
                    all_story_parts = []
                    current_context = ""
                    
                    for i in range(num_chunks):
                        # Modify system message for each chunk with strict word count requirements
                        chunk_min_words = max(min_chunk_words, chunk_size - 50)
                        chunk_max_words = min(max_chunk_words, chunk_size + 50)
                        
                        system_message = story_system_message.format(
                            min_words=chunk_min_words,
                            max_words=chunk_max_words
                        )
                        
                        # Create a specific prompt for each chunk that builds the story
                        if i == 0:
                            # First chunk: Set up the story and introduce characters
                            chunk_prompt = f"{user_prompt}\n\nWrite the beginning of the story. Introduce the main characters and set up the initial situation. This should be the first {chunk_size} words of the story."
                        elif i == num_chunks - 1:
                            # Last chunk: Resolve the story
                            context_parts = all_story_parts[-2:]  # Use last 2 chunks for context
                            current_context = " ".join(context_parts)
                            chunk_prompt = f"{user_prompt}\n\nPrevious story context: {current_context}\n\nWrite the final part of the story. Resolve the main conflicts and provide a satisfying conclusion. This should be the last {chunk_size} words of the story."
                        else:
                            # Middle chunks: Develop the story
                            context_parts = all_story_parts[-2:]  # Use last 2 chunks for context
                            current_context = " ".join(context_parts)
                            chunk_prompt = f"{user_prompt}\n\nPrevious story context: {current_context}\n\nContinue the story from where it left off. Develop the plot, add complications, and build tension. This should be the next {chunk_size} words of the story."
                        
                        # Generate this chunk
                        response = retrieval_chain.invoke({
                            "input": chunk_prompt,
                            "system_message": system_message
                        })
                        
                        chunk_content = response['answer']
                        # Replace any remaining original names with mapped names
                        chunk_content = replace_names_in_text(chunk_content, name_mapping)
                        all_story_parts.append(chunk_content)
                    
                    # Combine all chunks into final story
                    final_story = " ".join(all_story_parts)
                    # Final check to replace any remaining original names
                    final_story = replace_names_in_text(final_story, name_mapping)
                    
                    # Additional check for name consistency
                    for old_name, new_name in name_mapping.items():
                        if old_name in final_story:
                            print(f"Warning: Original name '{old_name}' still found in final story. Replacing with '{new_name}'.")
                            final_story = replace_names_in_text(final_story, {old_name: new_name})
                    
                    # Validate final word count
                    final_word_count = len(final_story.split())
                    if min_words <= final_word_count <= max_words:
                        # Use the vector store to get documents and scores
                        docs_with_scores = vectors.similarity_search_with_score(user_prompt, k=5)

                        elapsed_time = time.process_time() - start

                        # Save the generated story to MongoDB
                        generated_story = {
                            'prompt': user_prompt,
                            'story': final_story,
                            'response_time': elapsed_time,
                            'timestamp': datetime.now(),
                            'source_story_title': source_title,
                            'unique_id': unique_id,
                            'num_chunks': num_chunks,
                            'chunk_size': chunk_size,
                            'final_word_count': final_word_count,
                            'target_word_count': target_word_count,
                            'attempts': current_retry + 1
                        }
                        generated_stories_collection.insert_one(generated_story)

                        # Format the response
                        result = {
                            'answer': final_story,
                            'response_time': elapsed_time,
                            'similar_documents': [
                                {
                                    'content': doc.page_content,
                                    'similarity_score': float(score)  
                                } for doc, score in docs_with_scores
                            ]
                        }
                        
                        return result
                    else:
                        current_retry += 1
                        if current_retry >= max_retries:
                            # Instead of raising an error, return the last generated story
                            docs_with_scores = vectors.similarity_search_with_score(user_prompt, k=5)
                            elapsed_time = time.process_time() - start

                            # Save the generated story to MongoDB with a note about word count
                            generated_story = {
                                'prompt': user_prompt,
                                'story': final_story,
                                'response_time': elapsed_time,
                                'timestamp': datetime.now(),
                                'source_story_title': source_title,
                                'unique_id': unique_id,
                                'num_chunks': num_chunks,
                                'chunk_size': chunk_size,
                                'final_word_count': final_word_count,
                                'target_word_count': target_word_count,
                                'attempts': current_retry + 1,
                                'word_count_note': f'Generated {final_word_count} words (target: {min_words}-{max_words})'
                            }
                            generated_stories_collection.insert_one(generated_story)

                            # Format the response
                            result = {
                                'answer': final_story,
                                'response_time': elapsed_time,
                                'similar_documents': [
                                    {
                                        'content': doc.page_content,
                                        'similarity_score': float(score)  
                                    } for doc, score in docs_with_scores
                                ],
                                'word_count_note': f'Generated {final_word_count} words (target: {min_words}-{max_words})'
                            }
                            
                            return result
                        # Adjust chunk size for next attempt
                        if final_word_count < min_words:
                            chunk_size = int(chunk_size * 1.1)  # Increase chunk size by 10%
                        else:
                            chunk_size = int(chunk_size * 0.9)  # Decrease chunk size by 10%

                except Exception as e:
                    if current_retry >= max_retries:
                        # Instead of raising an error, return the last generated story if available
                        if 'final_story' in locals():
                            docs_with_scores = vectors.similarity_search_with_score(user_prompt, k=5)
                            elapsed_time = time.process_time() - start

                            # Save the generated story to MongoDB with error note
                            generated_story = {
                                'prompt': user_prompt,
                                'story': final_story,
                                'response_time': elapsed_time,
                                'timestamp': datetime.now(),
                                'source_story_title': source_title,
                                'unique_id': unique_id,
                                'num_chunks': num_chunks,
                                'chunk_size': chunk_size,
                                'final_word_count': len(final_story.split()),
                                'target_word_count': target_word_count,
                                'attempts': current_retry + 1,
                                'error_note': str(e)
                            }
                            generated_stories_collection.insert_one(generated_story)

                            # Format the response
                            result = {
                                'answer': final_story,
                                'response_time': elapsed_time,
                                'similar_documents': [
                                    {
                                        'content': doc.page_content,
                                        'similarity_score': float(score)  
                                    } for doc, score in docs_with_scores
                                ],
                                'error_note': str(e)
                            }
                            
                            return result
                        else:
                            # If no story was generated at all, raise the error
                            raise Exception(f"Error generating story: {str(e)}")
                    current_retry += 1
                    continue

    except Exception as e:
        raise Exception(f"Error generating story: {str(e)}")