from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import uuid
from datetime import datetime
import sys

# Add parent directory to path to import trad module
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import translation functions
from trad import get_blurbs, translate_blurb
import cv2

# Initialize FastAPI app
app = FastAPI(
    title="Open Translation API",
    description="Simple API for OCR-based image translation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Simple in-memory rate limiting (replace with Redis in production)
rate_limit_store = {}

# Simple API keys (replace with database in production)
API_KEYS = {
    "demo_key": {"rate_limit": 10, "plan": "demo"},
    "test_key": {"rate_limit": 100, "plan": "test"}
}

# Pydantic models
class TranslationResponse(BaseModel):
    task_id: str
    status: str
    translations: List[Dict[str, Any]]
    processing_time: float

# Rate limiting
async def check_rate_limit(api_key: str):
    """Simple rate limiting using in-memory storage"""
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    current_hour = datetime.now().strftime("%Y-%m-%d-%H")
    rate_key = f"rate_limit:{api_key}:{current_hour}"
    
    current_usage = rate_limit_store.get(rate_key, 0)
    if current_usage >= API_KEYS[api_key]["rate_limit"]:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    rate_limit_store[rate_key] = current_usage + 1
    
    # Clean up old entries (simple cleanup)
    if len(rate_limit_store) > 1000:
        old_keys = [k for k in rate_limit_store.keys() if not k.startswith("rate_limit:")]
        for k in old_keys[:100]:
            del rate_limit_store[k]

async def get_current_user(credentials = Depends(security)):
    """Validate API key"""
    api_key = credentials.credentials
    await check_rate_limit(api_key)
    return API_KEYS[api_key]

# Serve the main webpage
@app.get("/")
async def serve_index():
    return FileResponse("index.html")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Main translation endpoint
@app.post("/translate", response_model=TranslationResponse)
async def translate_image(
    file: UploadFile = File(...),
    input_language: str = Form(default="jpn_vert"),
    output_language: str = Form(default="en"),
    transparency: int = Form(default=200),
    ocr_mode: int = Form(default=5),
    current_user: dict = Depends(get_current_user)
):
    """Translate text from uploaded image"""
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    start_time = datetime.now()
    task_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file temporarily
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{task_id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process image
        img = cv2.imread(file_path)
        if img is None:
            raise Exception("Failed to load image")
        
        # Get text regions and translate
        try:
            blurbs, height, width = get_blurbs(img, input_language, ocr_mode)
        except Exception as e:
            # If get_blurbs fails, create a mock response with error info
            return TranslationResponse(
                task_id=task_id,
                status="error",
                translations=[{
                    "x": 0.0,
                    "y": 0.0,
                    "w": 1.0,
                    "h": 1.0,
                    "original_text": f"OCR Error: {str(e)}",
                    "translated_text": "Failed to process image",
                    "font_size": 12,
                    "error": True
                }],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
        
        translations = []
        
        for i, blurb in enumerate(blurbs):
            try:
                translated = translate_blurb(blurb, output_language)
                translation_info = {
                    "x": round(float(translated.x / width), 2),
                    "y": round(float(translated.y / height), 2),
                    "w": round(float(translated.w / width), 2),
                    "h": round(float(translated.h / height), 2),
                    "original_text": str(blurb.text),
                    "translated_text": str(translated.translation),
                    "font_size": round((translated.w * translated.h) / len(str(translated.translation))) if len(str(translated.translation)) > 0 else 12,
                    "blurb_index": i,
                    "error": False
                }
                translations.append(translation_info)
            except Exception as e:
                # Add error info for this specific blurb
                translation_info = {
                    "x": round(float(blurb.x / width), 2),
                    "y": round(float(blurb.y / height), 2),
                    "w": round(float(blurb.w / width), 2),
                    "h": round(float(blurb.h / height), 2),
                    "original_text": str(blurb.text),
                    "translated_text": f"Translation Error: {str(e)}",
                    "font_size": 12,
                    "blurb_index": i,
                    "error": True
                }
                translations.append(translation_info)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Clean up
        os.remove(file_path)
        
        return TranslationResponse(
            task_id=task_id,
            status="completed",
            translations=translations,
            processing_time=processing_time
        )
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

# Get supported languages
@app.get("/languages")
async def get_languages(current_user: dict = Depends(get_current_user)):
    return {
        "input_languages": {
            "japanese": "jpn_vert",
            "chinese": "chi_sim_vert", 
            "english": "eng",
            "korean": "kor",
            "spanish": "spa"
        },
        "output_languages": {
            "english": "en",
            "french": "fr",
            "german": "de",
            "spanish": "es",
            "italian": "it",
            "portuguese": "pt",
            "russian": "ru",
            "japanese": "ja",
            "chinese": "zh",
            "korean": "ko"
        }
    }

# Get usage stats
@app.get("/usage")
async def get_usage(current_user: dict = Depends(get_current_user)):
    current_hour = datetime.now().strftime("%Y-%m-%d-%H")
    # Use the API key from the current user context
    api_key = None
    for key, value in API_KEYS.items():
        if value == current_user:
            api_key = key
            break
    
    if api_key:
        rate_key = f"rate_limit:{api_key}:{current_hour}"
        current_usage = rate_limit_store.get(rate_key, 0)
        
        return {
            "current_usage": int(current_usage),
            "rate_limit": current_user["rate_limit"],
            "plan": current_user["plan"],
            "remaining": current_user["rate_limit"] - int(current_usage)
        }
    else:
        return {
            "current_usage": 0,
            "rate_limit": 0,
            "plan": "unknown",
            "remaining": 0
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port) 