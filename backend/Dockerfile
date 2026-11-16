# 1. Base image: a minimal Debian with python preinstalled
# 3.10 matches the .venv (Python 3.10.6), so packages behave the same.

FROM python:3.10-slim

# 2. Python settings that suit containers:
# -UNBUFFERED: print/log output shows up immediately in 'docker logs'
# -DONTWRITEBYTECODE: don't create .pyc files inside the image
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. ALl following commands run inside /app (created automatically).
WORKDIR /app

# 4. Copy Only the requirements file first, then install.
COPY  backend/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY backend/ ./backend/

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]