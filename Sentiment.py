from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.tokenize import sent_tokenize
import uvicorn
import spacy
import requests
import pickle
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import defaultdict
from heapq import nlargest
import json
from googleapiclient.discovery import build
import requests
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os
nltk.download('punkt')
load_dotenv()

API_KEY = os.getenv('API_KEY')
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
API_KEY_Location = os.getenv("API_KEY_Location")

app = FastAPI()

label_mapping = {'sentimental': 0, 'afraid': 1, 'proud': 2, 'faithful': 3, 'terrified': 4, 'joyful': 5, 'angry': 6, 'sad': 7, 'jealous': 8, 'grateful': 9, 'prepared': 10, 'embarrassed': 11, 'excited': 12, 'annoyed': 13, 'lonely': 14, 'ashamed': 15, 'guilty': 16, 'surprised': 17, 'nostalgic': 18, 'confident': 19, 'furious': 20, 'disappointed': 21, 'caring': 22, 'trusting': 23, 'disgusted': 24, 'anticipating': 25, 'anxious': 26, 'hopeful': 27, 'content': 28, 'impressed': 29, 'apprehensive': 30, 'devastated': 31}

model_name_or_path = "Sentiment_Model"  
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

reverse_label_mapping = {v: k for k, v in label_mapping.items()}

class SentimentRequest(BaseModel):
    text: str 


@app.post("/journal")
def predict_journal(request: SentimentRequest):

    nlp = spacy.load("en_core_web_sm")

    def preprocess_text(text):
        doc = nlp(text)
        tokens = [
            token.text
            for token in doc
            if token.text not in STOP_WORDS
            and token.text not in punctuation
            and token.pos_ not in ["PROPN", "VERB"]
        ]
        return " ".join(tokens)


    def keywords_text(text):
        doc = nlp(text)
        keywords = []

        for token in doc:
            if (
                token.text.lower() not in STOP_WORDS  
                and token.text not in punctuation  
                and len(token.text) > 2  
            ):
                if token.pos_ in ["NOUN", "ADJ", "PROPN"]:
                    keywords.append(token.lemma_)
                elif token.pos_ == "VERB" and token.dep_ in ["ROOT", "acl"]:
                    keywords.append(token.lemma_)

        for ent in doc.ents:
            if ent.text.lower() not in STOP_WORDS:
                keywords.append(ent.text)

        keyword_counts = Counter(keywords)
        ranked_keywords = [keyword for keyword, _ in keyword_counts.most_common(10)]  

        return ranked_keywords

    def calculate_similarity(sent1, sent2):
        doc1 = nlp(sent1)
        doc2 = nlp(sent2)
        return doc1.similarity(doc2)

    def summarize_text(text):
        preprocessed_text = preprocess_text(text)
        doc = nlp(preprocessed_text)
        sentences = [sent.text for sent in doc.sents]
        num_sentences = len(sentences)
        similarity_matrix = defaultdict(lambda: defaultdict(float))

        for i in range(num_sentences):
            for j in range(i + 1, num_sentences):
                similarity_matrix[i][j] = calculate_similarity(sentences[i], sentences[j])
                similarity_matrix[j][i] = similarity_matrix[i][j]

        scores = defaultdict(float)
        damping_factor = 0.85
        max_iter = 50
        convergence_threshold = 0.0001

        for _ in range(max_iter):
            prev_scores = scores.copy()
            for i in range(num_sentences):
                score = 1 - damping_factor
                for j in range(num_sentences):
                    if j != i:
                        score += (
                            damping_factor
                            * (similarity_matrix[i][j] / sum(similarity_matrix[j].values()))
                            * prev_scores[j]
                        )
                scores[i] = score

            if (
                sum(abs(scores[i] - prev_scores[i]) for i in range(num_sentences))
                < convergence_threshold
            ):
                break

        top_sentences = nlargest(3, scores, key=scores.get)
        summary = [sentences[i] for i in top_sentences]

        return " ".join(summary)

    def is_common_word(keyword):
        common_words = ["today", "tomorrow", "yesterday"]
        return keyword in common_words

    def is_date_related(keyword):
        doc = nlp(keyword)
        return any(ent.label_ == "DATE" for ent in doc.ents)

    #embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
    #embedding_model.save("Similarity_Model")

    embedding_model = SentenceTransformer("Similarity_Model")

    def get_latest_articles(keywords):
        service = build("customsearch", "v1", developerKey=API_KEY)

        refined_keywords = [
            keyword for keyword in keywords
            if not is_common_word(keyword) and not is_date_related(keyword)
        ]
        
        if not refined_keywords:
            return []

        query = " ".join(word for word in refined_keywords)  
        res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=10).execute()

        articles = []
        if "items" in res:
            for item in res["items"]:
                article = {
                    "title": item["title"],
                    "link": item["link"],
                    "snippet": item["snippet"],
                }
                articles.append(article)

        query_embedding = embedding_model.encode(" ".join(refined_keywords), convert_to_tensor=True)
        ranked_articles = []
        for article in articles:
            article_text = f"{article['title']} {article['snippet']}"
            article_embedding = embedding_model.encode(article_text, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(query_embedding, article_embedding).item()
            if similarity > 0.1:  
                ranked_articles.append((similarity, article))
        
        ranked_articles = sorted(ranked_articles, key=lambda x: x[0], reverse=True)
        return [article for _, article in ranked_articles[:5]]  

    def get_geocode(location):
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={API_KEY_Location}"
        response = requests.get(url).json()
        if "results" in response:
            result = response["results"][0]
            geometry = result["geometry"]
            location = geometry["location"]
            lat = location["lat"]
            lng = location["lng"]
            return f"{lat},{lng}"
        return None

    def get_place_type(keyword):
        place_type_mapping = {
            "museum": ["museum", "exhibit"],
            "restaurant": ["food", "cuisine", "restaurant", "dining"],
            "park": ["nature", "outdoors", "park", "garden"],
            "hotel": ["stay", "hotel", "resort"],
            "landmark": ["monument", "landmark", "sightseeing"],
            "shopping_mall": ["shopping", "mall", "retail"],
            "library": ["library", "books", "study"],
            "cafe": ["cafe", "coffee", "tea"]
        }
        for place_type, words in place_type_mapping.items():
            if any(word in keyword.lower() for word in words):
                return place_type
        return None 

    def get_nearby_places(keywords, location, limit=10):
        places = []
        geocode = get_geocode(location)
        if geocode:
            for keyword in keywords:
                if not is_common_word(keyword) and not is_date_related(keyword):
                    place_type = get_place_type(keyword)  
                    query_param = place_type if place_type else keyword 
                    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={geocode}&radius=5000&keyword={query_param}&key={API_KEY_Location}"
                    response = requests.get(url).json()
                    if "results" in response:
                        for result in response["results"]:

                            photo_reference = result.get("photos", [{}])[0].get("photo_reference", None)
                            photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={API_KEY_Location}" if photo_reference else None
                            place = {
                                "name": result["name"],
                                "address": result["vicinity"],
                                "rating": result.get("rating", None),
                                "types": result["types"],
                                "user_ratings_total": result.get("user_ratings_total", 0),
                                "image": photo_url,
                            }
                            places.append(place)
                            if len(places) >= limit:
                                return places
        places = sorted(places, key=lambda x: (-x["rating"] if x["rating"] else 0, -x["user_ratings_total"]))
        return places[:limit]

    input_text = request.text

    summary = summarize_text(input_text)

    summary_keywords = keywords_text(input_text)

    latest_articles = get_latest_articles(summary_keywords)

    location = "mumbai"
    nearby_places = get_nearby_places(summary_keywords, location)

    sentences = sent_tokenize(request.text)
    
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    
    predicted_labels = [reverse_label_mapping[label_id] for label_id in predictions.tolist()]
    
    result = {
        "input_text": input_text,
        "summary": summary,
        "keywords": summary_keywords,
        "latest_articles": latest_articles,
        "nearby_places": nearby_places,
        "sentences": sentences,
        "predictions": predicted_labels,
    }

    result_json = json.dumps(result, indent=2)
    return result_json


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  
    uvicorn.run(app, host="0.0.0.0", port=port)
