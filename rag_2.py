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
llm = ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'], model_name="gpt-4", temperature=0)

# System message for the LLM
system_message_template = """
You are a captivating digital storyteller creating binge-worthy content for YouTube audiences. Your mission is to transform Reddit stories into addictive, emotionally-charged narratives that keep viewers clicking. Follow these instructions:

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

6. **Word Count**:
   - Each story **must** be between {min_words}-{max_words} words. Do not generate responses outside this range.

7. **Content Strategy**:
   - Focus on high-drama scenarios: moral dilemmas, betrayals, unexpected kindness, etc.
   - Transform mundane Reddit stories by adding higher stakes and consequences
   - Create characters who make questionable decisions for understandable reasons
   - Incorporate elements that invite viewer discussion and commenting

Now, transform the Reddit story into a can't-look-away narrative that would thrive in today's attention economy. Make viewers feel something powerful enough that they'll want to share and comment.
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

    system_message = system_message_template.format(min_words=min_words, max_words=max_words)
    
    start = time.process_time()

    try:
        # Use the vector store to get documents and scores
        docs_with_scores = vectors.similarity_search_with_score(user_prompt, k=5)  # Retrieve top 5 documents

        response = retrieval_chain.invoke({"input": user_prompt, "system_message": system_message})
        elapsed_time = time.process_time() - start

        # Save the generated story to MongoDB
        generated_story = {
            'prompt': user_prompt,
            'story': response['answer'],
            'response_time': elapsed_time,
            'timestamp': datetime.now(),
            'source_story_title': source_title,
            'unique_id': unique_id
        }
        generated_stories_collection.insert_one(generated_story)

        # Format the response
        result = {
            'answer': response['answer'],
            'response_time': elapsed_time,
            'similar_documents': [
                {
                    'content': doc.page_content,
                    'similarity_score': float(score)  
                } for doc, score in docs_with_scores
            ]
        }
        
        return result

    except Exception as e:
        raise Exception(f"Error generating story: {str(e)}")