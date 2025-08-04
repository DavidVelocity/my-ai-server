from diffusers import DiffusionPipeline
from transformers import pipeline
import torch
import os

MODEL_DIR = "./models"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # <-- Pulls token from environment

def download_model(name, subfolder=None, is_pipeline=True, **kwargs):
    path = os.path.join(MODEL_DIR, subfolder if subfolder else name)
    if not os.path.exists(path):
        print(f"Downloading {name} to {path} ...")
        if is_pipeline:
            DiffusionPipeline.from_pretrained(
                name,
                use_auth_token=HF_TOKEN,  # <-- This line authenticates
                **kwargs
            ).save_pretrained(path)
        else:
            pipeline("text-to-speech", model=name, use_auth_token=HF_TOKEN).save_pretrained(path)
    else:
        print(f"Model {name} already downloaded.")

os.makedirs(MODEL_DIR, exist_ok=True)

download_model("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="t2v", torch_dtype=torch.float16)
download_model("stabilityai/stable-video-diffusion-img2vid-xt", subfolder="i2v", torch_dtype=torch.float16)
download_model("stabilityai/stable-diffusion-xl-base-1.0", subfolder="t2i", torch_dtype=torch.float16)
download_model("stabilityai/stable-diffusion-xl-refiner-1.0", subfolder="i2i", torch_dtype=torch.float16)
download_model("stabilityai/stable-audio-open-1.0", subfolder="tts", is_pipeline=False)