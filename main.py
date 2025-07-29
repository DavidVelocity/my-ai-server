from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import DiffusionPipeline
from transformers import pipeline as hf_pipeline
from PIL import Image
from io import BytesIO
import numpy as np
import imageio
import uuid
import os
import torch
import asyncio

# --- API Key ---
API_KEY = os.getenv("API_KEY", "changeme")  # Set this securely in your env

app = FastAPI()
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dancing-meerkat-6f06fa.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ASPECT_RATIOS = {
    "16:9": (1024, 576),
    "9:16": (576, 1024),
    "1:1": (768, 768),
    "4:3": (800, 600),
    "3:4": (600, 800),
}
DURATION_TO_FRAMES = {5: 30, 10: 60, 15: 90, 20: 120}

# --- Model placeholders ---
t2v_pipe = None
i2v_pipe = None
t2i_pipe = None
i2i_pipe = None
tts_pipe = None

@app.on_event("startup")
async def load_models():
    global t2v_pipe, i2v_pipe, t2i_pipe, i2i_pipe, tts_pipe
    print("Loading models...")

    t2v_pipe = DiffusionPipeline.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.float16
    ).to("cuda")

    i2v_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16
    ).to("cuda")

    t2i_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
    ).to("cuda")

    i2i_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16
    ).to("cuda")

    tts_pipe = hf_pipeline("text-to-speech", model="stabilityai/stable-audio-open-1.0")

    print("✅ All models loaded successfully.")

jobs = {}

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
    x_api_key: str = Header(None),
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    width, height = ASPECT_RATIOS.get(aspect_ratio, (512, 512))
    frames = DURATION_TO_FRAMES.get(duration, 30)
    uid = str(uuid.uuid4())

    if task_type == "text_to_image":
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        styled_prompt = f"{prompt}, {style or 'realistic'}"
        image = t2i_pipe(prompt=styled_prompt, width=width, height=height).images[0]
        filename = f"t2i-{uid}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        image.save(path)
        return {"output": {"image_url": f"/outputs/{filename}"}}

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
        return {"output": {"image_url": f"/outputs/{filename}"}}

    elif task_type == "text_to_video":
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        styled_prompt = f"{prompt}, {video_style or 'realistic'}"
        job_id = uid
        jobs[job_id] = {"status": "IN_QUEUE", "video_url": None}

        async def generate_video():
            jobs[job_id]["status"] = "IN_PROGRESS"
            video_frames = t2v_pipe(prompt=styled_prompt, width=width, height=height, num_frames=frames).frames[0]
            video_array = [np.array(f) for f in video_frames]
            filename = f"t2v-{job_id}.mp4"
            path = os.path.join(OUTPUT_DIR, filename)
            imageio.mimsave(path, video_array, fps=6)
            jobs[job_id]["status"] = "COMPLETED"
            jobs[job_id]["video_url"] = f"/outputs/{filename}"

        asyncio.create_task(generate_video())
        return {"id": job_id, "status": "IN_QUEUE"}

    elif task_type == "image_to_video":
        if not file or not prompt:
            raise HTTPException(status_code=400, detail="file and prompt are required")
        styled_prompt = f"{prompt}, {video_style or 'realistic'}"
        job_id = uid
        jobs[job_id] = {"status": "IN_QUEUE", "video_url": None}

        async def generate_video():
            jobs[job_id]["status"] = "IN_PROGRESS"
            image_data = await file.read()
            image = Image.open(BytesIO(image_data)).convert("RGB")
            image = image.resize((width, height))
            video_frames = i2v_pipe(image, prompt=styled_prompt, num_frames=frames).frames[0]
            video_array = [np.array(f) for f in video_frames]
            filename = f"i2v-{job_id}.mp4"
            path = os.path.join(OUTPUT_DIR, filename)
            imageio.mimsave(path, video_array, fps=6)
            jobs[job_id]["status"] = "COMPLETED"
            jobs[job_id]["video_url"] = f"/outputs/{filename}"

        asyncio.create_task(generate_video())
        return {"id": job_id, "status": "IN_QUEUE"}

    elif task_type == "text_to_speech":
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        audio = tts_pipe(prompt)
        filename = f"tts-{uid}.wav"
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "wb") as f:
            f.write(audio["audio"])
        return {"output": {"audio_url": f"/outputs/{filename}"}}

    else:
        raise HTTPException(status_code=400, detail="Unsupported task_type")

@app.get("/status/{job_id}")
async def status(job_id: str, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    response = {"id": job_id, "status": job["status"]}
    if job["status"] == "COMPLETED":
        response["output"] = {"video_url": job["video_url"]}
    return response

@app.get("/outputs/{filename}")
async def outputs(filename: str):
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
