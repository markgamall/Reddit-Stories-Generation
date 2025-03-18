import streamlit as st
from exa_modified import (
    generate_search_query,
    search_reddit_posts,
    fetch_subreddit_posts,
    fetch_post_by_url,
    save_to_file,
    generate_youtube_search_query,
    search_youtube_videos,
    reddit
)
from youtube_url import (
    process_youtube_url,
    get_video_id_from_url,
    get_video_info
)
from rag_2 import generate_story 
from datetime import datetime
import openai
import tempfile
import os
import requests
import sys
import json
import pyperclip
import traceback
import base64
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId
import contextlib
from gridfs import GridFS


# Load environment variables
load_dotenv()
openai_client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Initialize MongoDB connection
mongo_uri = os.environ['MONGO_URI']
client = MongoClient(mongo_uri)
db = client['reddit_stories_db']
fetched_stories_collection = db['fetched_stories']
generated_stories_collection = db['generated_stories']
youtube_transcriptions_collection = db['youtube_transcriptions']
fs = GridFS(db)











import subprocess

# Cache the ffmpeg installation check
@st.cache_resource
def setup_ffmpeg():
    try:
        # First check if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
        st.success("ffmpeg is already installed")
    except (FileNotFoundError, subprocess.CalledProcessError):
        st.warning("ffmpeg not found, installing alternative...")
        try:
            # Install imageio-ffmpeg (Python package) instead
            subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
            # Configure environment to use it
            os.environ["IMAGEIO_FFMPEG_EXE"] = os.path.join(sys.prefix, 'bin', 'ffmpeg')
            st.success("Installed imageio-ffmpeg as alternative")
        except Exception as e:
            st.error(f"Failed to install ffmpeg alternative: {e}")
            st.info("You may need to contact Streamlit support to enable ffmpeg for your deployment")

# Run this once at app startup
setup_ffmpeg()














# Initialize session state variables if they don't exist
if 'story_saved' not in st.session_state:
    st.session_state.story_saved = False
if 'current_story_title' not in st.session_state:
    st.session_state.current_story_title = None
if 'page' not in st.session_state:
    st.session_state.page = "Reddit Stories"
if 'error_handler_initialized' not in st.session_state:
    st.session_state.error_handler_initialized = False

def simulate_error():
    # This function will intentionally raise an error for testing
    raise ValueError("This is a simulated error for testing the bug reporting system.")

def log_error_to_db(error_message, error_type, traceback_info, source="Custom"):
    try:
        db.bug_reports.insert_one({
            'error_message': error_message,
            'error_type': error_type,
            'traceback': traceback_info,
            'source': source,
            'timestamp': datetime.now()
        })
        print(f"Error logged to database: {error_type} - {error_message}")
    except Exception as e:
        print(f"Failed to log error to MongoDB: {str(e)}")

# Global exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    error_message = str(exc_value)
    error_type = exc_type.__name__
    traceback_info = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    
    log_error_to_db(error_message, error_type, traceback_info, source="Global Exception")
    
    print(f"Unhandled exception: {error_message}")
    print(traceback_info)

# Set the exception handler
sys.excepthook = handle_exception

# Patch st.error to log errors to MongoDB when displayed
original_error = st.error

def patched_error(message, *args, **kwargs):
    # Log the error to MongoDB
    log_error_to_db(
        error_message=message,
        error_type="Streamlit UI Error",
        traceback_info="Error displayed via st.error()",
        source="st.error()"
    )
    # Call the original error function
    return original_error(message, *args, **kwargs)

# Apply the patch
st.error = patched_error

# Create a streamlit run-on-error handler for widgets
def handle_widget_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = str(e)
            error_type = type(e).__name__
            traceback_info = traceback.format_exc()
            
            log_error_to_db(
                error_message=error_message,
                error_type=error_type,
                traceback_info=traceback_info,
                source="Streamlit Widget"
            )
            
            # Re-raise to let Streamlit handle it normally
            raise
    return wrapper

# Patch common Streamlit widget functions to catch errors
for widget_func in ['button', 'checkbox', 'radio', 'selectbox', 'multiselect', 
                    'slider', 'select_slider', 'text_input', 'text_area',
                    'number_input', 'date_input', 'time_input']:
    if hasattr(st, widget_func):
        original_func = getattr(st, widget_func)
        setattr(st, widget_func, handle_widget_error(original_func))

# Add an error boundary around Streamlit app sections
class ErrorBoundary(contextlib.ContextDecorator):
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_message = str(exc_val)
            error_type = exc_type.__name__
            traceback_info = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            
            log_error_to_db(
                error_message=error_message,
                error_type=error_type,
                traceback_info=traceback_info,
                source="Streamlit Section"
            )
            
            # Don't suppress the exception
            return False
        return False

def generate_speech(text):
    """Generate narration audio with a specific voice style using OpenAI's TTS API"""
    try:
        client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        
        # Using OpenAI's Chat Completions API to generate audio with specific voice characteristics
        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],  # Output types that you want the model to generate
            audio={"voice": "alloy", "format": "mp3"},
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a helpful assistant that converts text into realistic audio speech.
                    
                    **Do not alter the text in any way. Read it as it is.**

                    For the audio, speak in a dramatic voice like a professional audiobook narrator. Emphasis on key moments.
                    Use natural inflections and an engaging storytelling style.
                    Speak clearly but with emotional expressiveness appropriate to the story content.
                    Vary your pacing for dramatic effect, slowing down at tense moments and speeding up during action.
                    
                    **Do not alter the text in any way. Read it as it is.**
                    """
                },
                {
                    "role": "user",
                    "content": text,
                }
            ],
        )
        
        # Extract the audio data
        audio_data = base64.b64decode(completion.choices[0].message.audio.data)
        
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        return temp_path
    except Exception as e: 
        st.error(f"Error generating speech: {str(e)}")
        log_error_to_db(str(e), type(e).__name__, traceback.format_exc())

        return None

def generate_speech_default(text):
    """
    Generate audio from text using OpenAI's TTS API with a custom voice description.
    
    Voice description: A young female voice with a warm, expressive tone that's engaging 
    and conversational. The voice has a slight touch of emotion, speaking clearly with good
    pacing and natural inflections. It sounds like a friendly storyteller who's genuinely 
    interested in sharing an exciting story.
    """
    try:
        # Using 'nova' voice which is a female voice that matches our description best
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        return temp_path
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        log_error_to_db(str(e), type(e).__name__, traceback.format_exc())

        return None

# Function to load a story from MongoDB to the text file
def load_story_to_file(story_id):
    story = fetched_stories_collection.find_one({"_id": ObjectId(story_id)})
    if story:
        content = f"Title: {story['title']}\n\nContent: {story['content']}"
        with open("reddit_story.txt", "w", encoding="utf-8") as file:
            file.write(content)
        
        # Debug: Print the content being written to the file
        st.write("Content written to file:")
        st.code(content)
        
        st.session_state.story_saved = True
        st.session_state.current_story_title = story['title']
        return True
    return False

# Create the sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page", 
    ["Reddit Stories", "YouTube Videos", "Story Generation"]
)

# Set the current page in session state
st.session_state.page = page

# Sidebar Instructions
st.sidebar.subheader("Instructions")
if page == "Reddit Stories":
    st.sidebar.markdown("""
    1. **Fetch a Reddit post** using one of the tabs.
    2. **Save the post** to use it as inspiration for story generation.
    3. **View your saved posts** in the fourth tab.
    4. Go to the Story Generation page when ready to create your story.
    """)
elif page == "YouTube Videos":
    st.sidebar.markdown("""
    1. **Fetch a YouTube video** by entering its URL or searching by query.
    2. The transcription will be automatically saved.
    3. **View your saved transcriptions** in the third tab.
    4. Go to the Story Generation page when ready to create your story.
    """)
else:  # Story Generation
    st.sidebar.markdown("""
    1. Make sure you have saved a Reddit post or YouTube transcription first.
    2. **Visit the Prompt Templates tab** to find or create the perfect prompt.
    3. **Go to the Generate Story tab** to create your story.
    4. **Enter a prompt** (paste one from the templates or write your own).
    5. **Generate and enjoy** your creative story!
    6. View your previously generated stories in the third tab.
    """)

# Main content based on selected page
if page == "Reddit Stories":
    st.title("Reddit Story Retrieval")
    
    # Create tabs for Reddit stories page
    tab1, tab2, tab3, tab4 = st.tabs([
        "Fetch by URL", "Fetch from Subreddit", "Fetch by Query", "View Fetched Stories"
    ])
    
    # Tab 1: Fetch by URL
    with tab1:
        st.header("Fetch Reddit Post by URL")
        url = st.text_input("Enter the Reddit thread URL:")
        
        if st.button("Fetch Post", key="fetch_post_by_url_button"):
            with st.spinner("Fetching post..."):
                post_content = fetch_post_by_url(url)
                save_to_file(f"Title: {post_content['title']}\n\nContent: {post_content['content']}")
                
                # Insert into MongoDB
                fetched_stories_collection.insert_one({
                    'title': post_content['title'],
                    'content': post_content['content'],
                    'source_url': url,
                    'timestamp': datetime.now()
                })
                
                # Update session state
                st.session_state.story_saved = True
                st.session_state.current_story_title = post_content['title']
                
                st.success(f"Saved the content of '{post_content['title']}' to file and MongoDB.")
                st.info("Go to the 'Story Generation' page to create a story based on this post!")
    
    # Tab 2: Fetch from Subreddit
    with tab2:
        st.header("Fetch Posts from Subreddit")
        subreddit_name = st.text_input("Enter the subreddit name (e.g., 'confession'):")
        sort_by = st.selectbox("Enter the sort method:", ["hot", "new", "top", "rising"])
        time_filter = "all"
        if sort_by == "top":
            time_filter = st.selectbox("Enter the time filter:", ["now", "today", "week", "month", "year", "all"])
        limit = st.slider("Enter the number of posts to fetch (up to 50):", 1, 50, 10)
        
        if st.button("Fetch Posts", key="fetch_subreddit_posts_button"):
            with st.spinner("Fetching posts..."):
                posts = fetch_subreddit_posts(subreddit_name, sort_by, time_filter, limit)
                st.session_state.posts = posts  # Store posts in session state
                st.write("Fetched Posts:")
                for i, post in enumerate(posts):
                    st.write(f"{i + 1}. {post['title']} (Score: {post['score']}, Comments: {post['num_comments']})")
                    with st.expander("View URL"):
                        st.write(post['url'])
        
        if "posts" in st.session_state:
            choice = st.number_input(
                "Choose a post to save (enter the number):", 
                1,  # min_value
                len(st.session_state.posts),  # max_value
                1,  # default_value
                key="subreddit_post_choice"  # Unique key for this number_input
            )
            if st.button("Save Selected Post", key="save_subreddit_post_button"):
                chosen_post = st.session_state.posts[choice - 1]
                save_to_file(f"Title: {chosen_post['title']}\n\nContent: {chosen_post['content']}")
                
                # Insert into MongoDB
                fetched_stories_collection.insert_one({
                    'title': chosen_post['title'],
                    'content': chosen_post['content'],
                    'source_url': chosen_post['url'],
                    'subreddit': subreddit_name,
                    'timestamp': datetime.now()
                })
                
                # Update session state
                st.session_state.story_saved = True
                st.session_state.current_story_title = chosen_post['title']
                
                st.success(f"Saved the content of '{chosen_post['title']}' to file and MongoDB.")
                st.info("Go to the 'Story Generation' page to create a story based on this post!")
    
    # Tab 3: Fetch by Query
    with tab3:
        st.header("Fetch Posts by Query")
        user_query = st.text_input("Enter your query (e.g., 'interesting cheating stories'):")
        num_results = st.slider("Enter the number of search results (up to 50):", 1, 50, 8)
        
        if st.button("Search Reddit", key="search_reddit_button"):
            with st.spinner("Searching Reddit..."):
                search_query = generate_search_query(user_query)
                st.write(f"Generated Search Query: {search_query}")
                
                search_results = search_reddit_posts(search_query, num_results)
                post_ids = [post['id'] for post in search_results]
                
                # Fetch detailed posts using PRAW
                detailed_posts = []
                for post_id in post_ids:
                    submission = reddit.submission(id=post_id)
                    submission.comments.replace_more(limit=0)
                    
                    post_metadata = {
                        "id": submission.id,
                        "title": submission.title,
                        "content": submission.selftext,
                        "score": submission.score,
                        "url": submission.url,
                        "subreddit": submission.subreddit.display_name,
                        "author": str(submission.author),
                        "created_utc": datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        "flair": submission.link_flair_text,
                        "num_comments": submission.num_comments,
                        "upvote_ratio": submission.upvote_ratio,
                        "post_length": len(submission.selftext),
                        "media": [],
                        "comments": []
                    }
                    detailed_posts.append(post_metadata)
                
                st.session_state.detailed_posts = detailed_posts  # Store posts in session state
                st.write("Fetched Posts:")
                for i, post in enumerate(detailed_posts):
                    st.write(f"{i + 1}. {post['title']} (Score: {post['score']}, Comments: {post['num_comments']})")
                    with st.expander("View URL"):
                        st.write(post['url'])
        
        if "detailed_posts" in st.session_state:
            if len(st.session_state.detailed_posts) > 0:  # Check if there are posts
                choice = st.number_input(
                    "Choose a post to save (enter the number):", 
                    1,  # min_value
                    len(st.session_state.detailed_posts),  # max_value
                    1,  # default_value
                    key="query_post_choice"  # Unique key for this number_input
                )
                if st.button("Save Selected Post", key="save_query_post_button"):
                    chosen_post = st.session_state.detailed_posts[choice - 1]
                    save_to_file(f"Title: {chosen_post['title']}\n\nContent: {chosen_post['content']}")
                    
                    # Insert into MongoDB
                    fetched_stories_collection.insert_one({
                        'title': chosen_post['title'],
                        'content': chosen_post['content'],
                        'source_url': chosen_post['url'],
                        'subreddit': chosen_post['subreddit'],
                        'query': user_query,
                        'timestamp': datetime.now()
                    })
                    
                    # Update session state
                    st.session_state.story_saved = True
                    st.session_state.current_story_title = chosen_post['title']
                    
                    st.success(f"Saved the content of '{chosen_post['title']}' to file and MongoDB.")
                    st.info("Go to the 'Story Generation' page to create a story based on this post!")
            else:
                st.warning("No posts fetched. Please try a different query.")
    
    # Tab 4: View Fetched Stories
    with tab4:
        st.header("View Fetched Stories")
        
        # Add a refresh button
        if st.button("Refresh Stories", key="refresh_fetched_stories"):
            st.rerun()
        
        # Get all fetched stories from MongoDB
        fetched_stories = list(fetched_stories_collection.find().sort("timestamp", -1))  # Sort by newest first
        
        if fetched_stories:
            st.info(f"Found {len(fetched_stories)} saved Reddit stories")
            
            # Display stories with expanders
            for story in fetched_stories:
                story_timestamp = story['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if 'timestamp' in story else "Unknown date"
                with st.expander(f"{story['title']} - {story_timestamp}"):
                    st.markdown(f"**Title:** {story['title']}")
                    st.markdown(f"**Source:** {story.get('source_url', 'Unknown')}")
                    st.markdown(f"**Subreddit:** {story.get('subreddit', 'Unknown')}")
                    st.markdown("**Content:**")
                    st.text_area("", story['content'], height=200, key=f"content_{story['_id']}", disabled=True)
                    
                    # Button to load this story for generation
                    if st.button(f"Load for Story Generation", key=f"load_{story['_id']}"):
                        if load_story_to_file(story['_id']):
                            st.success(f"Loaded '{story['title']}' for story generation.")
                            
                            # Add debug info to verify what was written to the file
                            with open("reddit_story.txt", "r", encoding="utf-8") as file:
                                file_content = file.read()
                            st.write("Content written to file for RAG:")
                            st.code(file_content)
                            
                            st.rerun()
        else:
            st.info("No fetched stories available. Use the first three tabs to fetch and save Reddit posts.")

elif page == "YouTube Videos":
    st.title("YouTube Video Transcriptions")
    
    # Create tabs for YouTube videos page
    tab1, tab2, tab3 = st.tabs([
        "Fetch YouTube Video", "Search YouTube Videos", "View YouTube Transcriptions"
    ])

    # Tab 1: Fetch YouTube Video (existing code)
    with tab1:
        st.header("Fetch YouTube Video Transcription")
        youtube_url = st.text_input("Enter the YouTube video URL:")
        
        if st.button("Fetch Video Transcription", key="fetch_youtube_button"):
            with st.spinner("Fetching video transcription... This may take a moment."):
                try:
                    # Pass the GridFS object to process_youtube_url
                    result = process_youtube_url(youtube_url, fs=fs)
                    
                    if result["success"]:
                        video_info = result["video_info"]
                        transcription = result["transcription"]
                        method = result["method"]
                        
                        # Insert into MongoDB
                        # Check if the YouTube URL already exists in the database
                        existing_video = youtube_transcriptions_collection.find_one({'source_url': youtube_url})
                        
                        if existing_video:
                            st.warning(f"The video '{video_info['title']}' is already saved in the database.")
                        else:
                            youtube_id = youtube_transcriptions_collection.insert_one({
                                'title': video_info['title'],
                                'channel': video_info['channel'],
                                'content': transcription,
                                'source_url': youtube_url,
                                'video_id': result["video_id"],
                                'transcription_method': method,
                                'duration': video_info['duration'],
                                'view_count': video_info['view_count'],
                                'upload_date': video_info['upload_date'],
                                'timestamp': datetime.now()
                            })
                        
                        st.success(f"Successfully fetched transcription for '{video_info['title']}' using {method}!")
                        
                        # Display video information
                        st.subheader("Video Information")
                        st.markdown(f"**Title:** {video_info['title']}")
                        st.markdown(f"**Channel:** {video_info['channel']}")
                        st.markdown(f"**Duration:** {video_info['duration']} seconds")
                        st.markdown(f"**View Count:** {video_info['view_count']}")
                        st.markdown(f"**Upload Date:** {video_info['upload_date']}")
                        
                        # Embed YouTube video
                        video_id = result["video_id"]
                        st.video(f"https://www.youtube.com/watch?v={video_id}")
                        
                        # Display transcription
                        st.subheader("Transcription")
                        st.text_area("", transcription, height=300)
                        
                        # Save to file for story generation
                        save_to_file(f"Title: {video_info['title']}\n\nContent: {transcription}")
                        
                        # Update session state
                        st.session_state.story_saved = True
                        st.session_state.current_story_title = video_info['title']
                        
                        st.info("Go to the 'Story Generation' page to create a story based on this transcription!")
                        
                    else:
                        st.error(f"Error: {result['error']}")
                        log_error_to_db(result['error'], "YouTube Fetch Error", traceback.format_exc())

                except Exception as e:
                    log_error_to_db(str(e), type(e).__name__, traceback.format_exc())
                    st.error(f"An error occurred: {str(e)}")

    # Tab 2: Search YouTube Videos (new code)
    with tab2:
        st.header("Search YouTube Videos by Query")
        user_query = st.text_input("Enter your search query (e.g., 'interesting tech stories'):")
        num_results = st.slider("Enter the number of search results (up to 10):", 1, 10, 5)
        
        if st.button("Search YouTube", key="search_youtube_button"):
            with st.spinner("Searching YouTube..."):
                try:
                    # Generate a refined search query using the LLM
                    search_query = generate_youtube_search_query(user_query)
                    st.write(f"Generated Search Query: {search_query}")
                    
                    # Search YouTube videos using Exa API
                    search_results = search_youtube_videos(search_query, num_results)
                    st.session_state.youtube_search_results = search_results
                    
                    if search_results:
                        st.write("Search Results:")
                        for i, video in enumerate(search_results):
                            st.write(f"{i + 1}. {video['title']} (Channel: {video['channel']}, Duration: {video['duration']}s)")
                            with st.expander("View URL"):
                                st.write(video['url'])
                    else:
                        st.warning("No videos found. Please try a different query.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    log_error_to_db(str(e), type(e).__name__, traceback.format_exc())
        
        if "youtube_search_results" in st.session_state:
            if len(st.session_state.youtube_search_results) > 0:
                choice = st.number_input(
                    "Choose a video to fetch (enter the number):", 
                    1,  # min_value
                    len(st.session_state.youtube_search_results),  # max_value
                    1,  # default_value
                    key="youtube_video_choice"  # Unique key for this number_input
                )
                if st.button("Fetch Selected Video", key="fetch_selected_youtube_video"):
                    chosen_video = st.session_state.youtube_search_results[choice - 1]
                    youtube_url = chosen_video['url']
                    
                    with st.spinner("Fetching video transcription... This may take a moment."):
                        try:
                            # Pass the GridFS object to process_youtube_url
                            result = process_youtube_url(youtube_url, fs=fs)
                            
                            if result["success"]:
                                video_info = result["video_info"]
                                transcription = result["transcription"]
                                method = result["method"]
                                
                                # Insert into MongoDB
                                # Check if the YouTube URL already exists in the database
                                existing_video = youtube_transcriptions_collection.find_one({'source_url': youtube_url})
                                
                                if existing_video:
                                    st.warning(f"The video '{video_info['title']}' is already saved in the database.")
                                else:
                                    youtube_id = youtube_transcriptions_collection.insert_one({
                                        'title': video_info['title'],
                                        'channel': video_info['channel'],
                                        'content': transcription,
                                        'source_url': youtube_url,
                                        'video_id': result["video_id"],
                                        'transcription_method': method,
                                        'duration': video_info['duration'],
                                        'view_count': video_info['view_count'],
                                        'upload_date': video_info['upload_date'],
                                        'timestamp': datetime.now()
                                    })
                                
                                st.success(f"Successfully fetched transcription for '{video_info['title']}' using {method}!")
                                
                                # Display video information
                                st.subheader("Video Information")
                                st.markdown(f"**Title:** {video_info['title']}")
                                st.markdown(f"**Channel:** {video_info['channel']}")
                                st.markdown(f"**Duration:** {video_info['duration']} seconds")
                                st.markdown(f"**View Count:** {video_info['view_count']}")
                                st.markdown(f"**Upload Date:** {video_info['upload_date']}")
                                
                                # Embed YouTube video
                                video_id = result["video_id"]
                                st.video(f"https://www.youtube.com/watch?v={video_id}")
                                
                                # Display transcription
                                st.subheader("Transcription")
                                st.text_area("", transcription, height=300)
                                
                                # Save to file for story generation
                                save_to_file(f"Title: {video_info['title']}\n\nContent: {transcription}")
                                
                                # Update session state
                                st.session_state.story_saved = True
                                st.session_state.current_story_title = video_info['title']
                                
                                st.info("Go to the 'Story Generation' page to create a story based on this transcription!")
                                
                            else:
                                st.error(f"Error: {result['error']}")
                                log_error_to_db(result['error'], "YouTube Fetch Error", traceback.format_exc())

                        except Exception as e:
                            log_error_to_db(str(e), type(e).__name__, traceback.format_exc())
                            st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("No videos fetched. Please try a different query.")

    # Tab 3: View YouTube Transcriptions (existing code)
    with tab3:
        st.header("View YouTube Transcriptions")
        
        # Add a refresh button
        if st.button("Refresh Transcriptions", key="refresh_youtube_transcriptions"):
            st.rerun()
        
        # Get all YouTube transcriptions from MongoDB
        youtube_transcriptions = list(youtube_transcriptions_collection.find().sort("timestamp", -1))  # Sort by newest first
        
        if youtube_transcriptions:
            st.info(f"Found {len(youtube_transcriptions)} saved YouTube transcriptions")
            
            # Display transcriptions with expanders
            for video in youtube_transcriptions:
                video_timestamp = video['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if 'timestamp' in video else "Unknown date"
                with st.expander(f"{video['title']} - {video_timestamp}"):
                    st.markdown(f"**Title:** {video['title']}")
                    st.markdown(f"**Channel:** {video.get('channel', 'Unknown')}")
                    st.markdown(f"**Source:** {video.get('source_url', 'Unknown')}")
                    st.markdown(f"**Transcription Method:** {video.get('transcription_method', 'Unknown')}")
                    
                    # Embed the video
                    if 'video_id' in video:
                        st.video(f"https://www.youtube.com/watch?v={video['video_id']}")
                    
                    st.markdown("**Transcription:**")
                    st.text_area("", video['content'], height=200, key=f"youtube_content_{video['_id']}", disabled=True)
                    
                    # Button to load this transcription for generation
                    if st.button(f"Load for Story Generation", key=f"load_youtube_{video['_id']}"):
                        save_to_file(f"Title: {video['title']}\n\nContent: {video['content']}")
                        
                        # Update session state
                        st.session_state.story_saved = True
                        st.session_state.current_story_title = video['title']
                        
                        st.success(f"Loaded '{video['title']}' for story generation.")
                        st.rerun()
        else:
            st.info("No YouTube transcriptions available. Use the 'Fetch YouTube Video' tab to fetch video transcriptions.")

else:  # Story Generation
    st.title("Story Generation")
    
    # Create tabs for Story Generation page
    tab1, tab2, tab3 = st.tabs([
        "Generate Story", "Prompt Templates", "View Generated Stories"
    ])
    
    # Tab 1: Generate Story
    with tab1:
        st.header("Generate a Story")
        
        # Check if a story has been loaded or saved
        if not st.session_state.story_saved:
            st.warning("No Reddit story or YouTube video has been selected yet. Please fetch and save content first from the Reddit Stories or YouTube Videos pages.")
        else:
            # Display information about the currently loaded story
            st.info(f"Currently loaded story: {st.session_state.current_story_title}")
            
            # Display the content of the saved story
            with open("reddit_story.txt", "r", encoding="utf-8") as file:
                saved_content = file.read()
            
            with st.expander("View Saved Content"):
                st.write(saved_content)
            
            st.subheader("Generate a creative story based on this content")
            user_prompt = st.text_area("Enter your prompt (e.g., 'Create a thriller based on this story'):", 
                                    height=100, 
                                    help="This prompt will guide the story generation based on the saved content.")
            
            min_words = st.number_input("Minimum number of words:", min_value=10, max_value=5000, value=500)
            max_words = st.number_input("Maximum number of words:", min_value=10, max_value=5000, value=1000)
            
            # Store generated content in session state
            if 'generated_story_text' not in st.session_state:
                st.session_state.generated_story_text = None
            if 'generated_audio_path' not in st.session_state:
                st.session_state.generated_audio_path = None
            if 'generation_result' not in st.session_state:
                st.session_state.generation_result = None
                
            if st.button("Generate Story", key="generate_story_button"):
                if not user_prompt:
                    st.error("Please enter a prompt for story generation.")
                else:
                    with st.spinner("Generating your story... This may take a moment."):
                        try:
                            # Call the generate_story function from rag_2.py
                            result = generate_story(user_prompt, min_words, max_words, st.session_state.current_story_title)
                            
                            # Store in session state
                            st.session_state.generated_story_text = result['answer']
                            st.session_state.generation_result = result
                            st.session_state.generated_audio_path = None  # Reset audio path
                            
                            # Save the generated story to MongoDB
                            generated_stories_collection.insert_one({
                                'source_story_title': st.session_state.current_story_title,
                                'prompt': user_prompt,
                                'story': result['answer'],
                                'response_time': result['response_time'],
                                'timestamp': datetime.now()
                            })

                        except Exception as e:
                            log_error_to_db(str(e), type(e).__name__, traceback.format_exc())
                            st.error(f"An error occurred: {str(e)}")
            
            # Display generated content if available in session state
            if st.session_state.generated_story_text:
                result = st.session_state.generation_result
                
                # Display the generated story
                st.subheader("Your Generated Story")
                st.markdown(st.session_state.generated_story_text)
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Response Time", f"{result['response_time']:.2f} seconds")
                
                # Optionally display similar documents used for generation
                with st.expander("View Source Passages Used"):
                    for i, doc in enumerate(result['similar_documents']):
                        st.markdown(f"**Passage {i+1}** (Similarity: {doc['similarity_score']:.4f})")
                        st.markdown(doc['content'])
                        st.divider()
                
                # Option to download the story
                st.download_button(
                    label="Download Story",
                    data=st.session_state.generated_story_text,
                    file_name="generated_story.txt",
                    mime="text/plain",
                    key="download_story_text"
                )

                st.subheader("Listen to Your Story")
                
                # Check if audio was already generated
                if st.session_state.generated_audio_path is None:
                    if st.button("Generate Audio Narration", key="generate_audio_button"):
                        with st.spinner("Generating audio narration..."):
                            try:
                                audio_path = generate_speech(st.session_state.generated_story_text)
                                if audio_path:
                                    st.session_state.generated_audio_path = audio_path
                                    st.success("Story narration ready! Listen below:")
                                else:
                                    st.error("Could not generate audio narration. Please try again.")
                                    log_error_to_db("Failed to generate audio narration", "Audio Generation Error", "")
                            except Exception as e:
                                st.error(f"Error generating speech: {str(e)}")
                                log_error_to_db(str(e), type(e).__name__, traceback.format_exc())
                
                # Display audio player and download button if audio exists
                if st.session_state.generated_audio_path:
                    st.audio(st.session_state.generated_audio_path)
                    
                    # Add download button for audio
                    with open(st.session_state.generated_audio_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        st.download_button(
                            label="Download Audio Narration",
                            data=audio_bytes,
                            file_name="story_narration.mp3",
                            mime="audio/mp3",
                            key="download_audio"
                        )
    
    # Tab 2: Prompt Templates
    # Tab 2: Prompt Templates (continued)
    with tab2:
        st.header("Story Prompt Templates")
        st.markdown("""
        Below are various prompt templates you can use in the "Generate Story" tab.
        Click the "Copy" button next to any prompt to copy it to your clipboard, then paste it in the prompt area of the Generate Story tab.
        """)
        
        # Define our predefined prompts with categories
        predefined_prompts = {
            "Genre-Specific": [
                {
                    "title": "Thriller/Mystery",
                    "prompt": "Transform this Reddit story into a gripping thriller with unexpected twists, red herrings, and a shocking revelation at the end. Include at least one character with questionable motives and create a sense of mounting tension throughout the narrative.",
                    "description": "Perfect for stories with elements of suspense or unexplained events"
                },
                {
                    "title": "Horror",
                    "prompt": "Rewrite this Reddit story as a psychological horror narrative that gradually builds dread. Focus on creating an atmosphere of unease and paranoia, with a disturbing revelation that changes everything. Use sensory details to create visceral discomfort.",
                    "description": "Works well with unsettling experiences or encounters"
                },
                {
                    "title": "Romance",
                    "prompt": "Turn this Reddit story into a heartfelt romance with emotional depth. Create a relationship arc with genuine obstacles, meaningful connection, and authentic character growth. Balance moments of tension with tender interactions.",
                    "description": "Great for relationship stories or missed connections"
                },
                {
                    "title": "Comedy",
                    "prompt": "Transform this Reddit story into a hilarious comedy with witty dialogue, comical misunderstandings, and absurd situations. Include humorous character quirks and build toward an entertaining payoff that surprises and delights.",
                    "description": "Best for awkward situations or unusual encounters"
                }
            ],
            "Narrative Styles": [
                {
                    "title": "First-Person Confession",
                    "prompt": "Rewrite this Reddit story as a raw, emotionally honest first-person confession. Include introspective moments where the narrator questions their actions and motivations. Focus on inner conflict and personal growth through difficult realizations.",
                    "description": "Ideal for personal experiences with moral complexity"
                },
                {
                    "title": "Multi-Perspective",
                    "prompt": "Retell this Reddit story from multiple perspectives, showing how different characters experienced and interpreted the same events. Reveal conflicting motivations, misunderstandings, and partial truths that create a complex narrative tapestry.",
                    "description": "Perfect for stories involving conflicts between multiple people"
                },
                {
                    "title": "Unreliable Narrator",
                    "prompt": "Transform this Reddit story using an unreliable narrator whose perspective gradually reveals itself to be flawed or biased. Plant subtle clues throughout that hint at the truth beneath their narrative, culminating in a revelation that forces readers to reinterpret earlier events.",
                    "description": "Works well with stories involving misunderstandings or conflicts"
                }
            ],
            "Emotional Focus": [
                {
                    "title": "Redemption Arc",
                    "prompt": "Reframe this Reddit story as a redemption narrative where the main character must confront their mistakes and work toward meaningful atonement. Focus on internal struggle, genuine growth, and the complex process of making amends.",
                    "description": "Best for stories involving regrets or mistakes"
                },
                {
                    "title": "Moral Dilemma",
                    "prompt": "Enhance this Reddit story by focusing on a central moral dilemma with no easy answers. Explore the complexity of the situation from multiple angles, showing how different values and priorities lead to difficult choices with significant consequences.",
                    "description": "Great for ethically complex situations"
                },
                {
                    "title": "Cathartic Resolution",
                    "prompt": "Transform this Reddit story into an emotional journey toward catharsis and healing. Build toward a powerful emotional climax where characters confront painful truths, release suppressed feelings, and find a measure of peace or closure.",
                    "description": "Perfect for emotional or unresolved situations"
                }
            ],
            "Creative Transformations": [
                {
                    "title": "Modern Fairy Tale",
                    "prompt": "Reimagine this Reddit story as a modern fairy tale with symbolic elements, archetypal characters, and an underlying moral lesson. Include subtle magical or surreal elements while maintaining emotional authenticity.",
                    "description": "Works with stories that have clear lessons or symbolic potential"
                },
                {
                    "title": "Epistolary Format",
                    "prompt": "Reconstruct this Reddit story as an epistolary narrative told through a series of letters, emails, text messages, social media posts, or journal entries. Use this format to reveal information gradually and show character perspectives in their own words.",
                    "description": "Interesting format for stories involving relationships or correspondence"
                },
                {
                    "title": "In Media Res Thriller",
                    "prompt": "Transform this Reddit story into a fast-paced thriller that starts in the middle of the action, then weaves in crucial backstory through flashbacks and dialogue. Create a ticking clock scenario that drives the narrative forward with urgency.",
                    "description": "Great for dramatic situations with high stakes"
                }
            ]
        }
        
        # Display prompts by category with copy buttons
        for category, prompts in predefined_prompts.items():
            st.subheader(category)
            
            for i, prompt_data in enumerate(prompts):
                with st.expander(f"{prompt_data['title']} - {prompt_data['description']}"):
                    st.markdown(f"**{prompt_data['title']}**")
                    st.markdown(f"_{prompt_data['description']}_")
                    st.text_area("Prompt:", prompt_data['prompt'], height=150, key=f"prompt_template_{category}_{i}", disabled=True)
                    
                    # Create a unique key for the copy button
                    copy_button_key = f"copy_button_{category}_{i}"
                    
                    # Add a copy button for each prompt
                    if st.button(
                        "📋 Copy Prompt", 
                        key=copy_button_key,
                        help="Copy this prompt to use in the Generate Story tab"
                    ):
                        # Copy the prompt text to the clipboard
                        pyperclip.copy(prompt_data['prompt'])
                        st.success("✅ Prompt copied to clipboard! You can now paste it in the Generate Story tab.")
        
        # Add a section for custom prompt creation
        st.subheader("Create Your Own Prompt Template")
        st.markdown("""
        Use the guidelines below to craft your own effective story prompts:
        
        1. **Be specific about tone and style**: Clearly state the emotional tone and writing style you want.
        2. **Define character development**: Specify how characters should evolve through the story.
        3. **Set narrative structure**: Request specific story elements (conflict, climax, resolution).
        4. **Request sensory details**: Ask for vivid descriptions that engage the senses.
        
        Experiment with different combinations to find what works best for your storytelling needs!
        """)
        
        # Add a section for prompt formulas
        st.subheader("Prompt Formula")
        st.markdown("""
        A good story prompt often follows this structure:
        
        ```
        Transform this Reddit story into [GENRE] with [SPECIFIC ELEMENT]. Focus on [NARRATIVE ASPECT] and include [SPECIAL FEATURE]. Create characters who [CHARACTER TRAIT/ACTION] and build toward [TYPE OF ENDING/RESOLUTION].
        ```
        
        Try combining elements from different templates to create your perfect prompt!
        """)

    # Tab 3: View Generated Stories
    with tab3:
        st.header("View Generated Stories")
        
        # Add a refresh button
        if st.button("Refresh Stories", key="refresh_generated_stories"):
            st.rerun()
        
        # Get all generated stories from MongoDB
        generated_stories = list(generated_stories_collection.find().sort("timestamp", -1))  # Sort by newest first
        
        if generated_stories:
            st.info(f"Found {len(generated_stories)} generated stories")
            
            # Display stories with expanders
            for story in generated_stories:
                story_timestamp = story['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if 'timestamp' in story else "Unknown date"
                story_title = f"Story generated on {story_timestamp}"
                
                if 'source_story_title' in story and story['source_story_title']:
                    story_title = f"Based on '{story['source_story_title']}' - {story_timestamp}"
                
                with st.expander(story_title):
                    st.markdown(f"**Prompt:** {story['prompt']}")
                    st.markdown(f"**Generated in:** {story.get('response_time', 'Unknown')} seconds")
                    st.markdown("**Story:**")
                    st.markdown(story['story'])
                    
                    # Download button for each story
                    st.download_button(
                        label="Download This Story",
                        data=story['story'],
                        file_name=f"generated_story_{story['_id']}.txt",
                        mime="text/plain",
                        key=f"download_{story['_id']}"
                    )
        else:
            st.info("No generated stories available. Use the 'Generate Story' tab to create stories based on Reddit posts or YouTube transcriptions.")