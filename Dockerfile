FROM python:3.11-slim

WORKDIR /app

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

EXPOSE 7860

CMD ["python", "-m", "src.app.app", "--host", "0.0.0.0", "--port", "7860", \
     "--model", "outputs/models/cnn1d_best.pt"]