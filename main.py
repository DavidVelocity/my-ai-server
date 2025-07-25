from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import DiffusionPipeline
from transformers import pipeline as hf_pipeline
from PIL import Image
import numpy as np
import imageio
import uuid
import os
import torch

app = FastAPI()
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Aspect Ratios and Duration Maps ---
ASPECT_RATIOS = {
    "16:9": (1024, 576),
    "9:16": (576, 1024),
    "1:1": (768, 768),
    "4:3": (800, 600),
    "3:4": (600, 800)
}
DURATION_TO_FRAMES = {5: 30, 10: 60, 15: 90, 20: 120}

# --- MODEL PIPELINES ---
# TEXT TO VIDEO
t2v_pipe = DiffusionPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.float16).to("cuda")

# IMAGE TO VIDEO
i2v_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16).to("cuda")

# TEXT TO IMAGE
t2i_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")

# IMAGE TO IMAGE
i2i_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16).to("cuda")

# TEXT TO SPEECH
tts_pipe = hf_pipeline("text-to-speech", model="stabilityai/stable-audio-open-1.0")

# --- ENDPOINTS ---

# 1. TEXT TO VIDEO
@app.post("/text-to-video")
async def text_to_video(
    prompt: str = Form(...),
    video_style: str = Form(...),
    aspect_ratio: str = Form("16:9"),
    duration: int = Form(5)
):
    styled_prompt = f"{prompt}, {video_style}"
    width, height = ASPECT_RATIOS.get(aspect_ratio, (512, 512))
    frames = DURATION_TO_FRAMES.get(duration, 30)
    video_frames = t2v_pipe(prompt=styled_prompt, width=width, height=height, num_frames=frames).frames[0]
    video_array = [np.array(f) for f in video_frames]
    filename = f"t2v-{uuid.uuid4()}.mp4"
    video_path = os.path.join(OUTPUT_DIR, filename)
    imageio.mimsave(video_path, video_array, fps=6)
    return FileResponse(video_path, media_type="video/mp4", filename=filename)


# 2. IMAGE TO VIDEO
@app.post("/image-to-video")
async def image_to_video(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    video_style: str = Form(...),
    aspect_ratio: str = Form("16:9"),
    duration: int = Form(5)
):
    styled_prompt = f"{prompt}, {video_style}"
    image = Image.open(await file.read()).convert("RGB")
    image = image.resize(ASPECT_RATIOS.get(aspect_ratio, (512, 512)))
    frames = DURATION_TO_FRAMES.get(duration, 30)
    video_frames = i2v_pipe(image, prompt=styled_prompt, num_frames=frames).frames[0]
    video_array = [np.array(f) for f in video_frames]
    filename = f"i2v-{uuid.uuid4()}.mp4"
    video_path = os.path.join(OUTPUT_DIR, filename)
    imageio.mimsave(video_path, video_array, fps=6)
    return FileResponse(video_path, media_type="video/mp4", filename=filename)


# 3. TEXT TO SPEECH
@app.post("/text-to-speech")
async def text_to_speech(
    prompt: str = Form(...),
    voice: str = Form("default")
):
    audio = tts_pipe(prompt)
    filename = f"tts-{uuid.uuid4()}.wav"
    audio_path = os.path.join(OUTPUT_DIR, filename)
    with open(audio_path, "wb") as f:
        f.write(audio["audio"])
    return FileResponse(audio_path, media_type="audio/wav", filename=filename)


# 4. TEXT TO IMAGE
@app.post("/text-to-image")
async def text_to_image(
    prompt: str = Form(...),
    style: str = Form("realistic"),
    aspect_ratio: str = Form("1:1")
):
    styled_prompt = f"{prompt}, {style}"
    width, height = ASPECT_RATIOS.get(aspect_ratio, (512, 512))
    image = t2i_pipe(prompt=styled_prompt, width=width, height=height).images[0]
    filename = f"t2i-{uuid.uuid4()}.png"
    image_path = os.path.join(OUTPUT_DIR, filename)
    image.save(image_path)
    return FileResponse(image_path, media_type="image/png", filename=filename)


# 5. IMAGE TO IMAGE
@app.post("/image-to-image")
async def image_to_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    transform_strength: str = Form("medium"),
    style: str = Form("artistic"),
    aspect_ratio: str = Form("1:1")
):
    styled_prompt = f"{prompt}, {style}"
    image = Image.open(await file.read()).convert("RGB")
    width, height = ASPECT_RATIOS.get(aspect_ratio, (512, 512))
    strength_map = {"low": 0.3, "medium": 0.5, "high": 0.7, "maximum": 0.9}
    strength = strength_map.get(transform_strength.lower(), 0.5)
    image = image.resize((width, height))
    output = i2i_pipe(prompt=styled_prompt, image=image, strength=strength).images[0]
    filename = f"i2i-{uuid.uuid4()}.png"
    image_path = os.path.join(OUTPUT_DIR, filename)
    output.save(image_path)
    return FileResponse(image_path, media_type="image/png", filename=filename)
