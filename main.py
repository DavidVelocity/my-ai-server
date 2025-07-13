from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableVideoDiffusionPipeline
import torch
import os
import uuid
import io
import numpy as np
import imageio
from PIL import Image
import torchaudio

app = FastAPI()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. TEXT-TO-IMAGE (T2I)
text2img_pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

@app.post("/generate-image")
async def generate_image(prompt: str = Form(...)):
    image = text2img_pipe(prompt=prompt).images[0]
    filename = f"{uuid.uuid4()}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    image.save(path)
    return FileResponse(path, media_type="image/png", filename=filename)


# 2. IMAGE-TO-IMAGE (I2I)
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

@app.post("/refine-image")
async def refine_image(prompt: str = Form(...), file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    result = img2img_pipe(prompt=prompt, image=image).images[0]
    filename = f"refined-{uuid.uuid4()}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    result.save(path)
    return FileResponse(path, media_type="image/png", filename=filename)


# 3. IMAGE-TO-VIDEO (I2V)
img2vid_pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

@app.post("/image-to-video")
async def image_to_video(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((512, 512))
    video_frames = img2vid_pipe(image, num_frames=14).frames[0]
    video_array = [np.array(f) for f in video_frames]

    filename = f"{uuid.uuid4()}.mp4"
    video_path = os.path.join(OUTPUT_DIR, filename)
    imageio.mimsave(video_path, video_array, fps=6, quality=8)
    return FileResponse(video_path, media_type="video/mp4", filename=filename)


# 4. TEXT-TO-VIDEO (T2V)
t2v_pipe = DiffusionPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    torch_dtype=torch.float16
).to("cuda")

@app.post("/text-to-video")
async def text_to_video(prompt: str = Form(...)):
    video_frames = t2v_pipe(prompt=prompt, num_inference_steps=40, num_frames=14).frames[0]
    video_array = [np.array(f) for f in video_frames]

    filename = f"t2v-{uuid.uuid4()}.mp4"
    video_path = os.path.join(OUTPUT_DIR, filename)
    imageio.mimsave(video_path, video_array, fps=6)
    return FileResponse(video_path, media_type="video/mp4", filename=filename)


# 5. TEXT-TO-SPEECH (TTS)
tts_model = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

@app.post("/text-to-speech")
async def text_to_speech(prompt: str = Form(...)):
    audio = tts_model(prompt).audios[0]

    filename = f"speech-{uuid.uuid4()}.wav"
    audio_path = os.path.join(OUTPUT_DIR, filename)
    torchaudio.save(audio_path, torch.tensor(audio), 44100)
    return FileResponse(audio_path, media_type="audio/wav", filename=filename)


@app.get("/")
def root():
    return {"message": "Model is ready!"}
