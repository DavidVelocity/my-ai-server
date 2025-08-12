FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN git submodule update --init --recursive

# Install system dependencies
RUN apt update && apt install -y python3 python3-pip git libgl1 libglib2.0-0

# Copy your code
WORKDIR /app
ARG GITHUB_TOKEN
RUN git clone --recurse-submodules https://${GITHUB_TOKEN}@github.com/DavidVelocity/my-ai-server.git .

# Pin pip for compatibility
RUN pip3 install "pip<25.3"

# Install PyTorch 2.6.0 with CUDA 12.1 first
RUN pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
RUN pip3 install --upgrade pip setuptools
RUN pip3 install --no-cache-dir -r requirements.txt

# Upgrade key libraries
RUN pip3 install --no-cache-dir --upgrade diffusers transformers tokenizers

# Install matching xformers
RUN pip install xformers==0.0.27.post2 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Hugging Face token
ARG HUGGINGFACE_TOKEN

# Download models during build
RUN python3 download_models.py

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
