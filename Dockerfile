FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 1. Install basic packages
RUN apt update && apt install -y python3 python3-pip git

# 2. Copy app
COPY . /app
WORKDIR /app

# 3. Install dependencies
RUN pip3 install --upgrade pip 
RUN pip3 install --no-cache-dir -r requirements.txt

# 4. Install xformers last (cleanest way)
RUN pip install xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu118

# 5. Launch FastAPI on RunPod-compatible port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
