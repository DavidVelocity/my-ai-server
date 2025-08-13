FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt update && apt install -y python3 python3-pip git libgl1 libglib2.0-0

WORKDIR /app
COPY . /app

# Upgrade pip and setuptools first
RUN pip3 install --upgrade pip setuptools

# Pin pip for compatibility (optional, you may skip this if upgrading above)
RUN pip3 install "pip<25.3"

# Install torch + torchvision + torchaudio with matching CUDA version
RUN pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.5.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install the rest of the dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Upgrade diffusers, transformers, tokenizers (don't upgrade torch here!)
RUN pip3 install --no-cache-dir --upgrade diffusers transformers tokenizers

# Install matching xformers
RUN pip install xformers==0.0.28.post2

# Make sure start.sh is executable
RUN chmod +x /app/start.sh

# Use start.sh as entrypoint
CMD ["./start.sh"]
