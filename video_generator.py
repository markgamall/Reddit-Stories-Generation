import os
import time
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Debug: Print all environment variables


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

class VideoGenerator:
    def __init__(self):
        self.api_key = os.getenv('HAILUO_API_KEY')
        if not self.api_key:
            raise ValueError("HAILUO_API_KEY not found in environment variables. Please check your .env file.")
        
        self.base_url = "https://api.minimaxi.chat/v1"
        self.headers = {
            'authorization': f'Bearer {self.api_key}',
            'content-type': 'application/json',
        }

    def invoke_video_generation(self, prompt: str, model: str = "T2V-01") -> str:
        """Submit a video generation task"""
        print("-----------------Submit video generation task-----------------")
        url = f"{self.base_url}/video_generation"
        payload = json.dumps({
            "prompt": prompt,
            "model": model
        })

        response = requests.request("POST", url, headers=self.headers, data=payload)
        print(response.text)
        task_id = response.json()['task_id']
        print("Video generation task submitted successfully, task ID: " + task_id)
        return task_id

    def query_video_generation(self, task_id: str):
        """Query the status of a video generation task"""
        url = f"{self.base_url}/query/video_generation?task_id={task_id}"
        response = requests.request("GET", url, headers=self.headers)
        status = response.json()['status']
        
        if status == 'Preparing':
            return "", 'Preparing'
        elif status == 'Queueing':
            return "", 'Queueing'
        elif status == 'Processing':
            return "", 'Processing'
        elif status == 'Success':
            return response.json()['file_id'], "Finished"
        elif status == 'Fail':
            return "", "Fail"
        else:
            return "", "Unknown"

    def fetch_video_result(self, file_id: str, output_file_name: str):
        """Download the generated video"""
        print("---------------Video generated successfully, downloading now---------------")
        url = f"{self.base_url}/files/retrieve?file_id={file_id}"
        response = requests.request("GET", url, headers=self.headers)
        print(response.text)

        download_url = response.json()['file']['download_url']
        print("Video download link: " + download_url)
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        
        with open(output_file_name, 'wb') as f:
            f.write(requests.get(download_url).content)
        print("The video has been downloaded to: " + output_file_name)
        return output_file_name

    def generate_video(self, prompt: str, output_file_name: str, model: str = "T2V-01"):
        """Complete video generation process"""
        try:
            # Submit the generation task
            task_id = self.invoke_video_generation(prompt, model)
            print("-----------------Video generation task submitted-----------------")

            # Poll for completion
            while True:
                time.sleep(10)
                file_id, status = self.query_video_generation(task_id)
                
                if file_id != "":
                    # Download the video
                    output_path = self.fetch_video_result(file_id, output_file_name)
                    print("---------------Successful---------------")
                    return output_path
                elif status == "Fail" or status == "Unknown":
                    print("---------------Failed---------------")
                    raise Exception(f"Video generation failed with status: {status}")

        except Exception as e:
            raise Exception(f"Error in video generation: {str(e)}") 