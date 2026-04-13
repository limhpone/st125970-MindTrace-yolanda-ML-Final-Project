# ── MindTrace · Emotion Prediction (AIT 2026) ────────────────────────────────
# Author: Aye Khin Khin Hpone (Yolanda Lim) · ST125970
#         Computer Science, Asian Institute of Technology

FROM python:3.11-slim

LABEL maintainer="st125970@ait.asia"
LABEL description="MindTrace — Emotion Prediction from Text"
LABEL author="Aye Khin Khin Hpone (Yolanda Lim)"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-download NLTK data at build time
RUN python -c "\
import nltk; \
nltk.download('stopwords', quiet=True); \
nltk.download('wordnet',   quiet=True); \
nltk.download('omw-1.4',   quiet=True)"

COPY app.py       .
COPY model.pkl    .
COPY templates/   templates/

EXPOSE 5000
ENV PORT=5000
ENV MODEL_PATH=model.pkl

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

CMD ["gunicorn","--bind","0.0.0.0:5000","--workers","2","--timeout","120","app:app"]
