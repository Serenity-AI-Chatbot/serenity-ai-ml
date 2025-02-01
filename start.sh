#!/bin/bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
gunicorn -k uvicorn.workers.UvicornWorker Sentiment:app --bind 0.0.0.0:$PORT
