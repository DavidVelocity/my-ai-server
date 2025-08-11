from diffusers import DiffusionPipeline
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq
import torch
import os
import subprocess

MODEL_DIR = "./models"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Pulls token from environment

def download_model(name, subfolder=None, is_pipeline=True, **kwargs):
    path = os.path.join(MODEL_DIR, subfolder if subfolder else name)
    if not os.path.exists(path):
        print(f"Downloading {name} to {path} ...")
        if is_pipeline:
            DiffusionPipeline.from_pretrained(
                name,
                token=HF_TOKEN,
                **kwargs
            ).save_pretrained(path)
        else:
            # For text-to-speech: download tokenizer and model explicitly
            AutoTokenizer.from_pretrained(
                name,
                token=HF_TOKEN,
                use_fast=False,
                cache_dir=path
            )
            AutoModelForSpeechSeq2Seq.from_pretrained(
                name,
                token=HF_TOKEN,
                cache_dir=path
            )
    else:
        print(f"Model {name} already downloaded.")

def download_coqui_xtts(repo_url="https://github.com/coqui-ai/xtts.git", subfolder="tts/xtts"):
    path = os.path.join(MODEL_DIR, subfolder)
    if not os.path.exists(path):
        print(f"Cloning Coqui XTTS from {repo_url} to {path} ...")
        subprocess.run(["git", "clone", repo_url, path], check=True)
    else:
        print("Coqui XTTS repo already cloned.")

os.makedirs(MODEL_DIR, exist_ok=True)

download_model("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="t2v", torch_dtype=torch.float16)
download_model("stabilityai/stable-video-diffusion-img2vid-xt", subfolder="i2v", torch_dtype=torch.float16)
download_model("stabilityai/stable-diffusion-xl-base-1.0", subfolder="t2i/base", torch_dtype=torch.float16)
download_model("stabilityai/stable-diffusion-xl-refiner-1.0", subfolder="t2i/refiner", torch_dtype=torch.float16)
download_model("stabilityai/stable-diffusion-xl-refiner-1.0", subfolder="i2i", torch_dtype=torch.float16)

# Instead, clone Coqui XTTS repo
download_coqui_xtts()