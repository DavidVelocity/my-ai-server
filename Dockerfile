FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 1. Install basic packages
RUN apt update && apt install -y python3 python3-pip git

# 2. Copy app files
COPY . /app
WORKDIR /app

# 3. Pin pip below 25.3 to avoid legacy build failures
RUN pip3 install "pip<25.3"

# 4. Install dependencies from requirements.txt
RUN pip3 install --upgrade pip setuptools && \
    pip3 install --no-cache-dir -r requirements.txt

# 5. Upgrade diffusers to latest version to avoid import errors
RUN pip3 install --no-cache-dir --upgrade diffusers

# 6. Upgrade transformers and tokenizers
RUN pip3 install --no-cache-dir --upgrade transformers tokenizers

# 7. Install xformers last (cleanest way)
RUN pip install xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu118

# 8. Pass Hugging Face token securely at build time
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

# 9. Download models once during image build
RUN python3 download_models.py

# 10. Run FastAPI app on RunPod port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
