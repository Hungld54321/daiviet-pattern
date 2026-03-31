FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi uvicorn open_clip_torch torch torchvision \
    pillow psycopg2-binary tqdm pandas python-multipart \
    diffusers transformers peft accelerate

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
