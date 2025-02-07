FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./
COPY Sentiment.py ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    && python -c "import nltk; nltk.download('punkt')" \
    && python -m nltk.downloader all

EXPOSE 5000

ENV HF_HOME=/app/huggingface_cache

CMD ["uvicorn", "Sentiment:app", "--host", "0.0.0.0", "--port", "5000"]
