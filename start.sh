#!/bin/bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
uvicorn Sentiment:app --host 0.0.0.0 --port 8000
