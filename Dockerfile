FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image much smaller
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN grep -v -i -E '^torch==' requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt

COPY app/ app/
COPY build_index.py .

# Placeholder key only for this step (config.py requires one to import).
# The real key is passed at runtime.
RUN GROQ_API_KEY=build-placeholder python build_index.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
