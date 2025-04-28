import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ImageGenerator:
    def __init__(self):
        self.api_key = os.getenv('HAILUO_API_KEY')
        if not self.api_key:
            raise ValueError("HAILUO_API_KEY not found in environment variables. Please check your .env file.")
        
        self.base_url = "https://api.minimaxi.chat/v1"
        self.headers = {
            'authorization': f'Bearer {self.api_key}',
            'content-type': 'application/json',
        }

    def generate_images(self, prompts: list, output_dir: str, aspect_ratio: str = "16:9", n: int = 1):
        """Generate images from a list of prompts"""
        try:
            all_image_urls = []
            
            for prompt in prompts:
                url = f"{self.base_url}/image_generation"
                payload = json.dumps({
                    "model": "image-01",
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "response_format": "url",
                    "n": n,
                    "prompt_optimizer": True
                })

                print(f"Sending request to {url}")
                print(f"Payload: {payload}")
                
                response = requests.request("POST", url, headers=self.headers, data=payload)
                print(f"Response status code: {response.status_code}")
                print(f"Response text: {response.text}")
                
                if response.status_code != 200:
                    raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
                
                result = response.json()
                
                # Check for API errors in base_resp
                if 'base_resp' in result and result['base_resp'].get('status_code') != 0:
                    error_msg = result['base_resp'].get('status_msg', 'Unknown error')
                    error_code = result['base_resp'].get('status_code', 'Unknown')
                    raise Exception(f"API Error (code {error_code}): {error_msg}")
                
                # Check if data is None
                if result.get('data') is None:
                    raise Exception("API returned null data. This might indicate an error or insufficient balance.")
                
                # Check if the response has the expected structure
                if not isinstance(result.get('data'), dict):
                    raise Exception(f"Unexpected API response format: {result}")
                
                if 'image_urls' not in result['data']:
                    raise Exception(f"API response missing 'image_urls' field: {result}")
                
                if result['data']['image_urls']:
                    all_image_urls.extend(result['data']['image_urls'])
                else:
                    print(f"Warning: No image URLs returned for prompt: {prompt}")
            
            if not all_image_urls:
                raise Exception("No image URLs were generated from any prompts")
            
            # Download all images
            downloaded_paths = []
            for i, image_url in enumerate(all_image_urls):
                image_filename = f"{output_dir}/generated_image_{i+1}.png"
                os.makedirs(output_dir, exist_ok=True)
                
                print(f"Downloading image {i+1} from URL: {image_url}")
                image_response = requests.get(image_url)
                if image_response.status_code != 200:
                    print(f"Warning: Failed to download image {i+1} from URL: {image_url}")
                    continue
                
                with open(image_filename, 'wb') as f:
                    f.write(image_response.content)
                downloaded_paths.append(image_filename)
                print(f"Successfully downloaded image {i+1} to: {image_filename}")
            
            if not downloaded_paths:
                raise Exception("Failed to download any images")
            
            return downloaded_paths
            
        except Exception as e:
            raise Exception(f"Error in image generation: {str(e)}")