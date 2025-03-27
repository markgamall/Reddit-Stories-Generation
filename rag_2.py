import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
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

# Load environment variables
load_dotenv()

# Initialize MongoDB connection
mongo_uri = os.environ['MONGO_URI']
client = MongoClient(mongo_uri)
db = client['reddit_stories_db']
generated_stories_collection = db['generated_stories']

# Initialize resources (global state)
embeddings = HuggingFaceEmbeddings()

# Initialize LLM and chains
llm = ChatOpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    model_name="gpt-4o",  # Using the latest GPT-4 model with largest context window
    temperature=0,  # Keeping low temperature for consistent output
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

2. NEVER include any of the following:
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
   - Freely change the number of characters to create relationship dynamics
   - Add or modify subplots that increase tension and stakes
   - Create secondary characters who challenge or complicate the protagonist's journey
   - Develop backstories that explain character motivations and conflicts
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

# Function to generate a story
def generate_story(user_prompt, min_words, max_words, source_title):
    # Load from the text file
    file_path = "reddit_story.txt"
    if not os.path.exists(file_path):
        raise Exception("No story content available")
    
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

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
            system_message = system_message_template.format(
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
            max_retries = 0  # Maximum number of retries if word count is off
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
                        
                        system_message = system_message_template.format(
                            min_words=chunk_min_words,
                            max_words=chunk_max_words
                        )
                        
                        # Add context from previous chunks if not the first chunk
                        if i > 0:
                            # For larger stories, use more context from previous chunks
                            context_chunks = min(3, i)  # Use up to 3 previous chunks for context
                            context_parts = all_story_parts[-context_chunks:]
                            current_context = " ".join(context_parts)
                            user_prompt_with_context = f"{user_prompt}\n\nPrevious story context: {current_context}"
                        else:
                            user_prompt_with_context = user_prompt
                        
                        # Generate this chunk
                        response = retrieval_chain.invoke({
                            "input": user_prompt_with_context,
                            "system_message": system_message
                        })
                        
                        chunk_content = response['answer']
                        all_story_parts.append(chunk_content)
                        
                        # Update context for next chunk
                        current_context = " ".join(all_story_parts[-2:])  # Use last 2 chunks as context
                    
                    # Combine all chunks into final story
                    final_story = " ".join(all_story_parts)
                    
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