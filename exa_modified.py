from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
import praw
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your API keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Initialize the LLM
llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=OPENAI_API_KEY)

# Initialize PRAW
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# Function to generate a search query using the LLM
def generate_search_query(user_query):
    prompt = PromptTemplate(
        input_variables=["user_query"],
        template="Generate a search query to find Reddit posts about: {user_query}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    search_query = chain.run(user_query)
    return search_query.strip()

# Function to search Reddit posts using Exa API
def search_reddit_posts(query, num_results=8):
    headers = {
        "Authorization": f"Bearer {EXA_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "query": f"{query} site:reddit.com",
        "num_results": num_results,
        "include_domains": ["reddit.com"]
    }
    response = requests.post(
        "https://api.exa.ai/search",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        raise Exception(f"Exa API request failed: {response.status_code}, {response.text}")
    
    search_results = response.json()
    extracted_data = []
    
    for result in search_results.get("results", []):
        url = result.get("url", "")
        if "reddit.com/r/" in url and "/comments/" in url:
            parts = url.split("/comments/")
            if len(parts) > 1:
                reddit_id = parts[1].split("/")[0]
                extracted_data.append({
                    "id": reddit_id,
                    "title": result.get("title", "Unknown Title"),
                    "url": url,
                    "snippet": result.get("text", "")
                })
    
    return extracted_data

# Function to fetch posts from a subreddit based on sorting and time filter
def fetch_subreddit_posts(subreddit_name, sort_by="hot", time_filter="all", limit=10):
    subreddit = reddit.subreddit(subreddit_name)
    
    if sort_by == "hot":
        posts = subreddit.hot(limit=limit)
    elif sort_by == "new":
        posts = subreddit.new(limit=limit)
    elif sort_by == "top":
        posts = subreddit.top(time_filter=time_filter, limit=limit)
    elif sort_by == "rising":
        posts = subreddit.rising(limit=limit)
    else:
        raise ValueError("Invalid sort_by parameter. Choose from 'hot', 'new', 'top', 'rising'.")
    
    post_list = []
    for post in posts:
        post_list.append({
            "id": post.id,
            "title": post.title,
            "url": post.url,
            "content": post.selftext,
            "score": post.score,
            "subreddit": post.subreddit.display_name,
            "author": str(post.author),
            "created_utc": datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            "flair": post.link_flair_text,
            "num_comments": post.num_comments,
            "upvote_ratio": post.upvote_ratio,
            "post_length": len(post.selftext),
            "media": [],
            "comments": []
        })
    
    return post_list

# Function to fetch a single post by URL
def fetch_post_by_url(url):
    post_id = url.split("/comments/")[1].split("/")[0]
    submission = reddit.submission(id=post_id)
    submission.comments.replace_more(limit=0)
    
    post_content = {
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
    return post_content

# Function to save the content of a single post to a file
def save_to_file(post_content, filename="reddit_story.txt"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(post_content)