from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import FileResponse
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
import soundfile as sf  # Moved import here, used in TTS

API_KEY = os.getenv("API_KEY", "changeme")  # Set securely in your environment
MODEL_DIR = "./models"
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

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

# Model placeholders
t2v_pipe = None  # Wan text-to-video pipeline
i2v_pipe = None  # StabilityAI image-to-video pipeline
t2i_base = None  # StabilityAI text-to-image base pipeline
t2i_refiner = None  # StabilityAI text-to-image refiner pipeline
i2i_pipe = None  # StabilityAI image-to-image pipeline

# TTS pipeline and speaker embedding
tts_pipe = None
speaker_embedding = None

@app.on_event("startup")
async def load_models():
    global t2v_pipe, i2v_pipe, t2i_base, t2i_refiner, i2i_pipe, tts_pipe, speaker_embedding
    print("Loading models from local disk...")

    # Wan text-to-video
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(MODEL_DIR, "t2v", "vae"), torch_dtype=torch.bfloat16
    )
    t2v_pipe = WanPipeline.from_pretrained(
        os.path.join(MODEL_DIR, "t2v"), vae=vae, torch_dtype=torch.bfloat16
    ).to("cuda")

    # StabilityAI image-to-video
    i2v_pipe = DiffusionPipeline.from_pretrained(
        os.path.join(MODEL_DIR, "i2v"), torch_dtype=torch.float16
    ).to("cuda")

    # StabilityAI text-to-image base and refiner
    t2i_base = DiffusionPipeline.from_pretrained(
        os.path.join(MODEL_DIR, "t2i", "base"),
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    t2i_base.to("cuda")

    t2i_refiner = DiffusionPipeline.from_pretrained(
        os.path.join(MODEL_DIR, "t2i", "refiner"),
        text_encoder_2=t2i_base.text_encoder_2,
        vae=t2i_base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    t2i_refiner.to("cuda")

    # StabilityAI image-to-image pipeline
    i2i_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        os.path.join(MODEL_DIR, "i2i"),
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    i2i_pipe.to("cuda")

    # Load Microsoft SpeechT5 TTS pipeline
    tts_pipe = hf_pipeline("text-to-speech", model="microsoft/speecht5_tts")

    # Load speaker embedding dataset and pick example embedding
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

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
            video_frames = i2v_pipe(image=image, prompt=styled_prompt).frames[0]
            filename = f"i2v-{job_id}.mp4"
            path = os.path.join(OUTPUT_DIR, filename)
            export_to_video(video_frames, path, fps=15)
            jobs[job_id]["status"] = "COMPLETED"
            jobs[job_id]["video_url"] = f"/outputs/{filename}"

        asyncio.create_task(generate_video())
        return {"id": job_id, "status": "IN_QUEUE"}

    elif task_type == "text_to_speech":
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        # Generate speech with microsoft/speecht5_tts pipeline and speaker embedding
        speech = tts_pipe(
            prompt,
            forward_params={"speaker_embeddings": speaker_embedding}
        )

        filename = f"tts-{uid}.wav"
        path = os.path.join(OUTPUT_DIR, filename)
        sf.write(path, speech["audio"], samplerate=speech["sampling_rate"])

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
