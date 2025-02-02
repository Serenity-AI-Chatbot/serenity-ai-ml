#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download xx_ent_wiki_sm
python -c "import nltk; nltk.download('punkt')"
