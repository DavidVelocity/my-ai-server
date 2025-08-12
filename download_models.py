from diffusers import DiffusionPipeline
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq
import torch
import os

MODEL_DIR = "./models"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Pull token from environment securely

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
            # For non-pipeline models, just download tokenizer and model explicitly
            download_tts_model(name, subfolder)
    else:
        print(f"Model {name} already downloaded.")

def download_tts_model(name, subfolder):
    path = os.path.join(MODEL_DIR, subfolder)
    if not os.path.exists(path):
        print(f"Downloading TTS model {name} to {path} ...")
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False, token=HF_TOKEN)
        tokenizer.save_pretrained(path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(name, token=HF_TOKEN)
        model.save_pretrained(path)
    else:
        print(f"TTS model {name} already downloaded.")

os.makedirs(MODEL_DIR, exist_ok=True)

# Image/Video models
download_model("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="t2v", torch_dtype=torch.float16)
download_model("stabilityai/stable-video-diffusion-img2vid-xt", subfolder="i2v", torch_dtype=torch.float16)
download_model("stabilityai/stable-diffusion-xl-base-1.0", subfolder="t2i/base", torch_dtype=torch.float16)
download_model("stabilityai/stable-diffusion-xl-refiner-1.0", subfolder="t2i/refiner", torch_dtype=torch.float16)
download_model("stabilityai/stable-diffusion-xl-refiner-1.0", subfolder="i2i", torch_dtype=torch.float16)

# Microsoft SpeechT5 TTS model only
download_tts_model("microsoft/speecht5_tts", "tts/speecht5")
