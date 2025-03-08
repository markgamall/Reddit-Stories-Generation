from flask import Flask, request, jsonify
import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
import time
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

# Load the OpenAI API key
openai_api_key = os.environ['OPENAI_API_KEY']

# Initialize MongoDB connection
mongo_uri = os.environ['MONGO_URI']
client = MongoClient(mongo_uri)
db = client['reddit_stories_db']
generated_stories_collection = db['generated_stories']

# Flask app initialization
app = Flask(__name__)

# Initialize resources (global state)
embeddings = HuggingFaceEmbeddings()

# Initialize LLM and chains
llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4", temperature=0)

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

3. **Emotional Impact**:
   - Design each story to trigger a specific emotional rollercoaster
   - Create flawed, complex characters readers can't help but care about
   - Include at least one moment that elicits a strong emotion (shock, outrage, heartbreak, etc.)
   - End with either justice/satisfaction OR a moral dilemma that sparks debate

4. **Conflict & Stakes**:
   - Every story must have conflict. Whether it's interpersonal, moral, or situational, the conflict should drive the narrative forward.
   - You have the freedom to change character names, subplots, or details to enhance the story's drama and relatability.
   - Feel free to add new elements or twists that weren't in the original Reddit story to make it more engaging.

4. **Word Count**:
   - Each story **must** be between {min_words}-{max_words} words. Do not generate responses outside this range.

5. **Content Strategy**:
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

# Define the persistent directory for Chroma
CHROMA_PERSIST_DIRECTORY = "chroma_db"

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({'status': 'ok'}), 200

@app.route('/ask', methods=['POST'])
def ask():
    """
    API endpoint to handle user prompts.
    Expects a JSON payload with the 'prompt', 'min_words', and 'max_words' fields.
    """
    # Parse the JSON input
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Invalid request, "prompt" field is required'}), 400

    user_prompt = data['prompt']
    min_words = data.get('min_words', 500)
    max_words = data.get('max_words', 1000)
    source_title = data.get('source_title', 'Unknown')

    # Load from the text file
    file_path = "reddit_story.txt"
    if not os.path.exists(file_path):
        return jsonify({'error': 'No story content available'}), 400
    
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Split the content into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.create_documents([content])

    # Generate a unique collection name for this request to avoid conflicts
    collection_name = f"reddit_stories_{uuid.uuid4().hex}"
    
    # Initialize a fresh Chroma DB with the current story
    vectors = Chroma.from_documents(
        documents=final_documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    
    # Create the retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    system_message = system_message_template.format(min_words=min_words, max_words=max_words)
    
    start = time.process_time()

    try:
        # Use the Chroma vector store directly to get documents and scores
        docs_with_scores = vectors.similarity_search_with_score(user_prompt, k=5)  # Retrieve top 5 documents

        # Verify documents in the vector store (for debugging)
        print(f"Collection name: {collection_name}")
        print("Documents in the vector store:")
        db_docs = vectors.get()['documents']
        print(f"Number of documents: {len(db_docs)}")
        for i, doc in enumerate(db_docs):
            print(f"Doc {i+1}: {doc[:100]}...")  # Print first 100 chars of each document

        response = retrieval_chain.invoke({"input": user_prompt, "system_message": system_message})
        elapsed_time = time.process_time() - start

        # Save the generated story to MongoDB
        generated_story = {
            'prompt': user_prompt,
            'story': response['answer'],
            'response_time': elapsed_time,
            'timestamp': datetime.now(),
            'source_story_title': source_title,
            'collection_name': collection_name  # Store the collection name for reference
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
        
        # Optionally, you could try to delete the collection after use,
        # but we won't attempt this to avoid permission errors
        # Instead, we'll use a unique collection name each time
        
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)