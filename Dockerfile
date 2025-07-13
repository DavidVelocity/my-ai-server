FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 1. Update system and install basic packages
RUN apt update && apt install -y python3 python3-pip git

# 2. Copy your app code to the container
COPY . /app
WORKDIR /app

# 3. Install all packages listed in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# 4. Separately install xformers AFTER the above line
RUN pip install xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu118

# 5. Start your FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
