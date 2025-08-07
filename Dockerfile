FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt update && apt install -y python3 python3-pip git libgl1 libglib2.0-0

COPY . /app
WORKDIR /app

RUN pip3 install "pip<25.3"

# Install torch and torchaudio first (matching CUDA version)
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install all other dependencies except torch and torchaudio
RUN pip3 install --upgrade pip setuptools
RUN pip3 install --no-cache-dir -r requirements.txt

# Upgrade diffusers, transformers, and tokenizers
RUN pip3 install --no-cache-dir --upgrade diffusers transformers tokenizers

# Install xformers last
RUN pip install xformers --extra-index-url https://download.pytorch.org/whl/cu118

ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

RUN python3 download_models.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
