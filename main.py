from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import (
    DiffusionPipeline,
    WanPipeline,
    AutoencoderKLWan,
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.utils import load_image, export_to_video
from transformers import pipeline as hf_pipeline
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import numpy as np
import imageio
import uuid
import os
import torch
import asyncio
import soundfile as sf 
import logging
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = os.getenv("API_KEY", "changeme")
MODEL_DIR = "./models"
OUTPUT_DIR = "./outputs"
OUTPUT_BASE_URL = os.getenv("OUTPUT_BASE_URL", "http://localhost:8000")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Velocity AI Backend", version="1.0.0")

# Updated CORS to match project requirements
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Configure via environment variable
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Aspect ratios and duration mappings
ASPECT_RATIOS = {
    "16:9": (1024, 576),
    "9:16": (576, 1024),
    "1:1": (768, 768),
    "4:3": (800, 600),
    "3:4": (600, 800),
}
DURATION_TO_FRAMES = {5: 30, 10: 60, 15: 90, 20: 120}

# Model placeholders
t2v_pipe = None
i2v_pipe = None
t2i_base = None
t2i_refiner = None
i2i_pipe = None
tts_pipe = None
speaker_embedding = None

@app.on_event("startup")
async def load_models():
    global t2v_pipe, i2v_pipe, t2i_base, t2i_refiner, i2i_pipe, tts_pipe, speaker_embedding
    logger.info("Loading models from local disk...")

    try:
        # Wan text-to-video
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(MODEL_DIR, "t2v", "vae"), torch_dtype=torch.bfloat16
        )
        t2v_pipe = WanPipeline.from_pretrained(
            os.path.join(MODEL_DIR, "t2v"),
            vae=vae,
            torch_dtype=torch.bfloat16,
            enable_xformers_memory_efficient_attention=False
        ).to("cuda")
        t2v_pipe.enable_attention_slicing()

        # StabilityAI image-to-video
        i2v_pipe = DiffusionPipeline.from_pretrained(
            os.path.join(MODEL_DIR, "i2v"), 
            torch_dtype=torch.float16,
            enable_xformers_memory_efficient_attention=False
        ).to("cuda")
        i2v_pipe.enable_attention_slicing()

        # StabilityAI text-to-image base
        t2i_base = DiffusionPipeline.from_pretrained(
            os.path.join(MODEL_DIR, "t2i", "base"),
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            enable_xformers_memory_efficient_attention=False
        ).to("cuda")
        t2i_base.enable_attention_slicing()

        # StabilityAI text-to-image refiner
        t2i_refiner = DiffusionPipeline.from_pretrained(
            os.path.join(MODEL_DIR, "t2i", "refiner"),
            text_encoder_2=t2i_base.text_encoder_2,
            vae=t2i_base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            enable_xformers_memory_efficient_attention=False
        ).to("cuda")
        t2i_refiner.enable_attention_slicing()

        # StabilityAI image-to-image
        i2i_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            os.path.join(MODEL_DIR, "i2i"),
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            enable_xformers_memory_efficient_attention=False
        ).to("cuda")
        i2i_pipe.enable_attention_slicing()

        # Load Microsoft SpeechT5 TTS pipeline
        tts_pipe = hf_pipeline("text-to-speech", model="microsoft/speecht5_tts")

        # Load speaker embedding dataset
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        logger.info("✅ All models loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        raise

# In-memory job storage (consider using Redis or database for production)
jobs = {}

# Middleware for API key validation
@app.middleware("http")
async def validate_api_key(request: Request, call_next):
    # Skip validation for health check and outputs endpoints
    if request.url.path in ["/", "/health"] or request.url.path.startswith("/outputs/"):
        return await call_next(request)
    
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"success": False, "error": "Unauthorized - Invalid API key"}
        )
    
    return await call_next(request)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Velocity AI Backend",
        "version": "1.0.0",
        "models_loaded": all([t2v_pipe, i2v_pipe, t2i_base, t2i_refiner, i2i_pipe, tts_pipe])
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models": {
            "text_to_video": t2v_pipe is not None,
            "image_to_video": i2v_pipe is not None,
            "text_to_image_base": t2i_base is not None,
            "text_to_image_refiner": t2i_refiner is not None,
            "image_to_image": i2i_pipe is not None,
            "text_to_speech": tts_pipe is not None,
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/run")
async def run(
    task_type: str = Form(...),
    prompt: str = Form(None),
    style: str = Form(None),
    file: UploadFile = File(None),
    video_style: str = Form(None),
    aspect_ratio: str = Form("1:1"),
    duration: int = Form(5),
    transform_strength: str = Form("medium"),
):
    """
    Main generation endpoint compatible with Bolt Database Edge Function proxy
    """
    try:
        width, height = ASPECT_RATIOS.get(aspect_ratio, (768, 768))
        frames = DURATION_TO_FRAMES.get(duration, 30)
        uid = str(uuid.uuid4())

        logger.info(f"Processing {task_type} request with ID: {uid}")

        if task_type == "text_to_image":
            if not prompt:
                raise HTTPException(status_code=400, detail="prompt is required")
            
            styled_prompt = f"{prompt}, {style or 'realistic'}"
            
            image_latent = t2i_base(
                prompt=styled_prompt,
                num_inference_steps=40,
                denoising_end=0.8,
                output_type="latent",
                height=height,
                width=width,
            ).images
            
            image = t2i_refiner(
                prompt=styled_prompt,
                num_inference_steps=40,
                denoising_start=0.8,
                image=image_latent,
            ).images[0]
            
            filename = f"t2i-{uid}.png"
            path = os.path.join(OUTPUT_DIR, filename)
            image.save(path)
            
            return {
                "success": True,
                "image_url": f"{OUTPUT_BASE_URL}/outputs/{filename}",
                "task_type": task_type,
                "job_id": uid
            }

        elif task_type == "image_to_image":
            if not file or not prompt:
                raise HTTPException(status_code=400, detail="file and prompt are required")
            
            styled_prompt = f"{prompt}, {style or 'artistic'}"
            image_data = await file.read()
            image = Image.open(BytesIO(image_data)).convert("RGB")
            
            strength_map = {"low": 0.3, "medium": 0.5, "high": 0.7, "maximum": 0.9}
            strength = strength_map.get(transform_strength.lower(), 0.5)
            
            image = image.resize((width, height))
            output = i2i_pipe(prompt=styled_prompt, image=image, strength=strength).images[0]
            
            filename = f"i2i-{uid}.png"
            path = os.path.join(OUTPUT_DIR, filename)
            output.save(path)
            
            return {
                "success": True,
                "image_url": f"{OUTPUT_BASE_URL}/outputs/{filename}",
                "task_type": task_type,
                "job_id": uid
            }

        elif task_type == "text_to_video":
            if not prompt:
                raise HTTPException(status_code=400, detail="prompt is required")
            
            styled_prompt = f"{prompt}, {video_style or 'realistic'}"
            job_id = uid
            jobs[job_id] = {
                "status": "IN_QUEUE",
                "video_url": None,
                "task_type": task_type,
                "created_at": datetime.utcnow().isoformat()
            }

            async def generate_video():
                try:
                    jobs[job_id]["status"] = "IN_PROGRESS"
                    logger.info(f"Starting video generation for job {job_id}")
                    
                    video_frames = t2v_pipe(
                        prompt=styled_prompt,
                        negative_prompt="",
                        height=height,
                        width=width,
                        num_frames=frames,
                        guidance_scale=5.0,
                    ).frames[0]
                    
                    video_array = [np.array(f) for f in video_frames]
                    filename = f"t2v-{job_id}.mp4"
                    path = os.path.join(OUTPUT_DIR, filename)
                    imageio.mimsave(path, video_array, fps=15)
                    
                    jobs[job_id]["status"] = "COMPLETED"
                    jobs[job_id]["video_url"] = f"{OUTPUT_BASE_URL}/outputs/{filename}"
                    logger.info(f"Video generation completed for job {job_id}")
                except Exception as e:
                    logger.error(f"Error in video generation for job {job_id}: {str(e)}")
                    jobs[job_id]["status"] = "FAILED"
                    jobs[job_id]["error"] = str(e)

            asyncio.create_task(generate_video())
            
            return {
                "success": True,
                "id": job_id,
                "status": "IN_QUEUE",
                "message": "Video generation started"
            }

        elif task_type == "image_to_video":
            if not file or not prompt:
                raise HTTPException(status_code=400, detail="file and prompt are required")
            
            styled_prompt = f"{prompt}, {video_style or 'realistic'}"
            job_id = uid
            jobs[job_id] = {
                "status": "IN_QUEUE",
                "video_url": None,
                "task_type": task_type,
                "created_at": datetime.utcnow().isoformat()
            }

            async def generate_video():
                try:
                    jobs[job_id]["status"] = "IN_PROGRESS"
                    logger.info(f"Starting image-to-video generation for job {job_id}")
                    
                    image_data = await file.read()
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                    image = image.resize((width, height))
                    
                    video_frames = i2v_pipe(image=image, prompt=styled_prompt).frames[0]
                    filename = f"i2v-{job_id}.mp4"
                    path = os.path.join(OUTPUT_DIR, filename)
                    export_to_video(video_frames, path, fps=15)
                    
                    jobs[job_id]["status"] = "COMPLETED"
                    jobs[job_id]["video_url"] = f"{OUTPUT_BASE_URL}/outputs/{filename}"
                    logger.info(f"Image-to-video generation completed for job {job_id}")
                except Exception as e:
                    logger.error(f"Error in image-to-video generation for job {job_id}: {str(e)}")
                    jobs[job_id]["status"] = "FAILED"
                    jobs[job_id]["error"] = str(e)

            asyncio.create_task(generate_video())
            
            return {
                "success": True,
                "id": job_id,
                "status": "IN_QUEUE",
                "message": "Image-to-video generation started"
            }

        elif task_type == "text_to_speech":
            if not prompt:
                raise HTTPException(status_code=400, detail="prompt is required")
            
            speech = tts_pipe(
                prompt,
                forward_params={"speaker_embeddings": speaker_embedding}
            )
            
            filename = f"tts-{uid}.wav"
            path = os.path.join(OUTPUT_DIR, filename)
            sf.write(path, speech["audio"], samplerate=speech["sampling_rate"])
            
            return {
                "success": True,
                "audio_url": f"{OUTPUT_BASE_URL}/outputs/{filename}",
                "task_type": task_type,
                "job_id": uid
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported task_type: {task_type}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in run endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}")
async def status(job_id: str):
    """
    Check job status - compatible with Bolt Database Edge Function proxy
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
        "success": True,
        "id": job_id,
        "status": job["status"],
        "task_type": job.get("task_type"),
        "created_at": job.get("created_at")
    }
    
    if job["status"] == "COMPLETED":
        response["output"] = {"video_url": job["video_url"]}
        response["video_url"] = job["video_url"]
    elif job["status"] == "FAILED":
        response["error"] = job.get("error", "Unknown error")
    
    return response

@app.get("/outputs/{filename}")
async def outputs(filename: str):
    """
    Serve generated files
    """
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    media_type = "application/octet-stream"
    if filename.endswith(".mp4"):
        media_type = "video/mp4"
    elif filename.endswith(".png"):
        media_type = "image/png"
    elif filename.endswith(".wav"):
        media_type = "audio/wav"
    
    return FileResponse(file_path, media_type=media_type, filename=filename)

@app.get("/voices")
async def get_voices():
    """
    Return available TTS voices
    Compatible with frontend expectations
    """
    # Return mock voices data - customize based on your TTS model
    voices = [
        {
            "id": "default",
            "name": "Default Voice",
            "description": "Standard neutral voice",
            "category": "general"
        }
    ]
    
    return {
        "success": True,
        "voices": voices,
        "count": len(voices)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
