
import streamlit as st
from exa_modified import (
    generate_search_query,
    search_reddit_posts,
    fetch_subreddit_posts,
    fetch_post_by_url,
    save_to_file,
    reddit
)
from datetime import datetime
import os
import requests
import json
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId
from rag_2 import generate_story  # Import the function from rag_2.py

# Load environment variables
load_dotenv()

# Initialize MongoDB connection
mongo_uri = os.environ['MONGO_URI']
client = MongoClient(mongo_uri)
db = client['reddit_stories_db']
fetched_stories_collection = db['fetched_stories']
generated_stories_collection = db['generated_stories']

# Initialize session state variables if they don't exist
if 'story_saved' not in st.session_state:
    st.session_state.story_saved = False
if 'current_story_title' not in st.session_state:
    st.session_state.current_story_title = None

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

# Streamlit UI
st.title("Reddit Story Generator")

# Create six tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Fetch by URL", "Fetch from Subreddit", "Fetch by Query", "Generate Story", "View Fetched Stories", "View Generated Stories"])

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
            st.info("Go to the 'Generate Story' tab to create a story based on this post!")

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
        choice = st.number_input("Choose a post to save (enter the number):", 1, len(st.session_state.posts), 1)
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
            st.info("Go to the 'Generate Story' tab to create a story based on this post!")

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
                1  # default_value
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
                st.info("Go to the 'Generate Story' tab to create a story based on this post!")
        else:
            st.warning("No posts fetched. Please try a different query.")


# Tab 4: Generate Story
with tab4:
    st.header("Generate a Story from Reddit Post")
    
    # Check if a story has been loaded or saved
    if not st.session_state.story_saved:
        st.warning("No Reddit story has been selected yet. Please fetch and save a post first from one of the other tabs, or select a story from the 'View Fetched Stories' tab.")
    else:
        # Display information about the currently loaded story
        st.info(f"Currently loaded story: {st.session_state.current_story_title}")
        
        # Display the content of the saved story
        with open("reddit_story.txt", "r", encoding="utf-8") as file:
            saved_content = file.read()
        
        with st.expander("View Saved Reddit Content"):
            st.write(saved_content)
        
        st.subheader("Generate a creative story based on this Reddit post")
        user_prompt = st.text_area("Enter your prompt (e.g., 'Create a thriller based on this story'):", 
                                   height=100, 
                                   help="This prompt will guide the story generation based on the saved Reddit content.")
        
        min_words = st.number_input("Minimum number of words:", min_value=10, max_value=5000, value=500)
        max_words = st.number_input("Maximum number of words:", min_value=10, max_value=5000, value=1000)
        
        if st.button("Generate Story", key="generate_story_button"):
            if not user_prompt:
                st.error("Please enter a prompt for story generation.")
            else:
                with st.spinner("Generating your story... This may take a moment."):
                    try:
                        # Call the generate_story function from rag_2.py
                        result = generate_story(user_prompt, min_words, max_words, st.session_state.current_story_title)
                        
                        # Display the generated story
                        st.subheader("Your Generated Story")
                        st.markdown(result['answer'])
                        
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
                        story_text = result['answer']
                        st.download_button(
                            label="Download Story",
                            data=story_text,
                            file_name="generated_story.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

# Tab 5: View Fetched Stories
with tab5:
    st.header("View Fetched Stories")
    
    # Add a refresh button
    if st.button("Refresh Stories", key="refresh_fetched_stories"):
        st.rerun()  # Changed from st.experimental_rerun()
    
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

# Tab 6: View Generated Stories
with tab6:
    st.header("View Generated Stories")
    
    # Add a refresh button
    if st.button("Refresh Stories", key="refresh_generated_stories"):
        st.rerun()  # Changed from st.experimental_rerun()
    
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
        st.info("No generated stories available. Use the 'Generate Story' tab to create stories based on Reddit posts.")

st.sidebar.subheader("Instructions")
st.sidebar.markdown(
    """
    1. **Fetch a Reddit post** using one of the first three tabs
    2. **Save the post** to use it as inspiration
    3. **Go to the Generate Story tab** to create your story
    4. **Enter a prompt** to guide the story generation
    5. **Generate and enjoy** your creative story!
    
    You can also:
    - View and load your previously saved Reddit posts
    - Browse your previously generated stories
    """
)