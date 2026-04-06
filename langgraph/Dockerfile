FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (includes agent/ and tests/)
COPY . .

# Set environment variables
ENV PYTHONPATH="/app"
ENV PYTHONUTF8=1

EXPOSE 2024

CMD ["python", "agent/main.py"]
