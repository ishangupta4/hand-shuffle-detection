FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY . .

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["python", "-m", "src.app.app", "--host", "0.0.0.0", "--port", "7860", \
     "--model", "outputs/models/cnn1d_best.pt"]
```

**2. `requirements-app.txt` (add at repo root)**

This is separate from your training `requirements.txt` — only what the server needs at inference time, with CPU-only PyTorch to keep the image lean:
```
fastapi==0.115.0
uvicorn==0.30.0
python-multipart==0.0.9
opencv-python-headless==4.10.0.84
numpy==1.26.4
scipy==1.13.0
mediapipe==0.10.14
torch==2.3.1+cpu
--extra-index-url https://download.pytorch.org/whl/cpu