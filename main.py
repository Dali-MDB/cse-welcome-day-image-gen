from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import json
import os 
import io
import requests
from dotenv import load_dotenv

load_dotenv()

with open("token_track.json", "r") as f:
    data = json.load(f)

current_token = data.get("current_token", 1) 
API_KEY = os.getenv(f'HF_TOKEN{current_token}')

API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-dev"

def api_call(prompt: str, key: str):
    """Generate image and return bytes directly"""
    headers = {"Authorization": f"Bearer {key}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    
    if response.status_code != 200:
        raise Exception(f"API call failed with status {response.status_code}")
    
    return response.content


# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post('/generate')
@limiter.limit("2/minute")
async def generate_image(request: Request, prompt: str):
    global API_KEY
    
    try:
        # Try with current token
        image_bytes = api_call(prompt, API_KEY)
        
    except Exception as e:
        print(f"Token expired or error: {e}")
        
        # Read current token index from JSON
        with open("token_track.json", "r") as f:
            data = json.load(f)

        current_token = data.get("current_token", 1)
        next_token = current_token + 1

        # Update the file with the new token index
        data["current_token"] = next_token
        with open("token_track.json", "w") as f:
            json.dump(data, f)

        # Get new API key and retry
        API_KEY = os.getenv(f'HF_TOKEN{next_token}')
        
        if not API_KEY:
            raise HTTPException(status_code=500, detail="No more tokens available")
        
        try:
            image_bytes = api_call(prompt, API_KEY)
        except Exception as retry_error:
            raise HTTPException(status_code=500, detail=f"Image generation failed: {str(retry_error)}")

    # Return image directly without saving
    return StreamingResponse(
        io.BytesIO(image_bytes),
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=img.png"}
    )



@app.get('/')
async def avoid_idle():
    return {'gg':'gg'}