FROM python:3.12-slim

WORKDIR /app

# Install build dependencies and TA-Lib C library
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    tar \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && apt-get purge -y --auto-remove build-essential wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --upgrade python-binance

# Copy the entire backend source code into /app
# Structure should be preserved to allow 'import backend.xxx'
COPY . /app

# Set PYTHONPATH to include /app so 'import backend' works
ENV PYTHONPATH=/app

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8001"]
