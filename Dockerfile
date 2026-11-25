FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Never run the server as root inside the container.
RUN useradd --create-home appuser
USER appuser

EXPOSE 4000

# PORT is provided by hosting platforms such as Render; defaults to 4000.
# --proxy-headers lets the rate limiter see the real client IP behind the platform's proxy.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-4000} --proxy-headers --forwarded-allow-ips='*'"]
