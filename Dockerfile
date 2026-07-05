FROM python:3.11-slim

WORKDIR /app

# Torch CPU-only (Hugging Face Spaces bepul tier GPU bermaydi, CPU wheel ~10x kichik)
COPY requirements-api.txt .
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY config/ config/
COPY results/models/cnn_best.pth results/models/cnn_best.pth
COPY main.py .

# HF Spaces konteynerni root bo'lmagan user bilan ishga tushirishi mumkin
ENV HOME=/tmp

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
