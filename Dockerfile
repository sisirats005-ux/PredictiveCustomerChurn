# Production-grade Dockerfile for ConnectTel Churn Predictor API

# Stage 1: Build dependencies and wheels
FROM python:3.11-slim as builder

WORKDIR /build

# Install compiler dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Serve the application
FROM python:3.11-slim as runner

WORKDIR /app

# Create an unprivileged runtime user for safer container execution.
RUN addgroup --system app && adduser --system --ingroup app app

# Copy installed site-packages from builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy all project code
COPY . /app/

# Expose port 8000 for FastAPI API, 8501 for Streamlit app
EXPOSE 8000
EXPOSE 8501

USER app

# Default environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Run FastAPI API by default
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
