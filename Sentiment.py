from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import nltk
from nltk.tokenize import sent_tokenize
import uvicorn
import spacy
import requests
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
#from huggingface_hub import hf_hub_download
from huggingface_hub import login
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from twilio.twiml.messaging_response import MessagingResponse
import numpy as np
from loguru import logger 
import random
from telegram import Bot
import telegram
import httpx


nltk.download('punkt')
load_dotenv()

API_KEY = os.getenv('API_KEY')
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
API_KEY_Location = os.getenv("API_KEY_Location")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
#BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
WEBHOOK_URL = "https://serenity-ai-ml.onrender.com/telegram/webhook"

bot = Bot(token=TELEGRAM_BOT_TOKEN)

#bot.set_webhook(WEBHOOK_URL)

#client = httpx.AsyncClient()

hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

app = FastAPI()

label_mapping = {'sentimental': 0, 'afraid': 1, 'proud': 2, 'faithful': 3, 'terrified': 4, 'joyful': 5, 'angry': 6, 'sad': 7, 'jealous': 8, 'grateful': 9, 'prepared': 10, 'embarrassed': 11, 'excited': 12, 'annoyed': 13, 'lonely': 14, 'ashamed': 15, 'guilty': 16, 'surprised': 17, 'nostalgic': 18, 'confident': 19, 'furious': 20, 'disappointed': 21, 'caring': 22, 'trusting': 23, 'disgusted': 24, 'anticipating': 25, 'anxious': 26, 'hopeful': 27, 'content': 28, 'impressed': 29, 'apprehensive': 30, 'devastated': 31}

reverse_label_mapping = {v: k for k, v in label_mapping.items()}

class SentimentRequest(BaseModel):
    text: str 

@app.get("/")
def read_root():
    return {"Hello": "World"}

# URL of the Node backend (adjust host/port as needed)
NODE_BACKEND_URL = os.getenv("NODE_BACKEND_URL")

def send_message(chat_id, text):
    # url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # payload = {"chat_id": chat_id, "text": text}
    # requests.post(url, json=payload)
    try:
        bot.send_message(chat_id=chat_id, text=text)
    except Exception as e:
        logger.error(f"Failed to send message: {e}")

@app.post("/telegram/webhook")
async def telegram_webhook(req: Request, background_tasks: BackgroundTasks):

    logger.info("Message Received")
    try:
        data = await req.json()
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
        return JSONResponse(content={"status": "error"}, status_code=200)

    background_tasks.add_task(process_update, data)
    return JSONResponse(content={"status": "OK"}, status_code=200)

def process_update(data):

    update = telegram.Update.de_json(data, bot)
    logger.info(update)
    
    if update.message and update.message.text:
        chat_id = update.message.chat.id
        user_message = update.message.text
        
        try:
            # here i am calling our node backend for gemini ka response :)
            node_response = requests.post(
                NODE_BACKEND_URL,
                json={"message": user_message, "from": chat_id}
            )
            node_response.raise_for_status()
            response_data = node_response.json()
            bot_response = response_data.get("response", "Sorry, something went wrong.")
            send_message(chat_id, bot_response)

        except Exception as e:
            logger.error(f"Error calling Node backend: {e}")
            send_message(chat_id, "Sorry, something went wrong.")
        
@app.post("/journal")
def predict_journal(request: SentimentRequest):

    model_name_or_path = "dhruvbcodes/Sentiment_Model"  
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

    embedding_model = SentenceTransformer("dhruvbcodes/Similarity_Model")

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

    #song pred:

    df = pd.read_csv('Datasets/dataset.csv')

    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo']

    data = df[features]

    mood_mapping = {
    'sentimental': {'valence': 0.6, 'acousticness': 0.4},
    'afraid': {'energy': 0.5, 'loudness': 0.4, 'speechiness': 0.3},
    'proud': {'energy': 0.7, 'danceability': 0.5, 'valence': 0.6},
    'faithful': {'acousticness': 0.6, 'valence': 0.5},
    'terrified': {'energy': 0.8, 'loudness': 0.7},
    'joyful': {'danceability': 0.7, 'valence': 0.8},
    'angry': {'energy': 0.9, 'loudness': 0.9},
    'sad': {'acousticness': 0.7, 'valence': 0.2},
    'jealous': {'energy': 0.6, 'speechiness': 0.5},
    'grateful': {'acousticness': 0.6, 'valence': 0.7},
    'prepared': {'energy': 0.6, 'tempo': 0.7},
    'embarrassed': {'speechiness': 0.6, 'loudness': 0.5},
    'excited': {'danceability': 0.9, 'energy': 0.8},
    'annoyed': {'loudness': 0.7, 'speechiness': 0.6},
    'lonely': {'acousticness': 0.7, 'valence': 0.3},
    'ashamed': {'acousticness': 0.6, 'speechiness': 0.5},
    'guilty': {'acousticness': 0.6, 'valence': 0.4},
    'surprised': {'tempo': 0.8, 'energy': 0.6},
    'nostalgic': {'acousticness': 0.8, 'valence': 0.5},
    'confident': {'energy': 0.7, 'danceability': 0.6},
    'furious': {'energy': 0.9, 'loudness': 0.9},
    'disappointed': {'valence': 0.3, 'acousticness': 0.6},
    'caring': {'acousticness': 0.7, 'valence': 0.6},
    'trusting': {'valence': 0.7, 'acousticness': 0.5},
    'disgusted': {'speechiness': 0.7, 'loudness': 0.6},
    'anticipating': {'tempo': 0.7, 'energy': 0.6},
    'anxious': {'energy': 0.6, 'speechiness': 0.5},
    'hopeful': {'valence': 0.8, 'danceability': 0.7},
    'content': {'valence': 0.9, 'acousticness': 0.6},
    'impressed': {'energy': 0.7, 'valence': 0.8},
    'apprehensive': {'speechiness': 0.6, 'energy': 0.5},
    'devastated': {'acousticness': 0.8, 'valence': 0.1, 'danceability': 0.2},
    }

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    model = KMeans()

    kmeans = KMeans(n_clusters=32, random_state=42)
    df['cluster'] = kmeans.fit_predict(data_scaled)

    centroids = kmeans.cluster_centers_

    '''
    def match_mood_to_centroid(centroid, mood_thresholds):
        score = 0
        for feature, threshold in mood_thresholds.items():
            feature_idx = features.index(feature)
            score += abs(centroid[feature_idx] - threshold)
        return score

    cluster_moods = {}
    for i, centroid in enumerate(centroids):
        scores = {mood: match_mood_to_centroid(centroid, thresholds) 
                for mood, thresholds in mood_mapping.items()}
        best_mood = min(scores.items(), key=lambda x: x[1])[0]
        cluster_moods[i] = best_mood

    def generate_song_for_mood(mood):
        if mood not in cluster_moods.values():
            return "Mood not found"
        for k, v in cluster_moods.items():
            if v == mood:
                cluster_id = k
                break
        song_features = centroids[cluster_id] 
        song_attr = scaler.inverse_transform(song_features.reshape(1, -1))[0]  
        distances = cdist(df[features], song_attr.reshape(1, -1), metric='euclidean').flatten()
        closest_song_idx = distances[:5]
        closest_song = df.iloc[closest_song_idx].sort_values('popularity', ascending=False).head(1)
        return closest_song
    '''

    def match_mood_to_centroid(centroid, mood_thresholds):
    # Initialize weighted score
        weighted_score = 0
        
        # Define feature weights (you can adjust these)
        feature_weights = {
            'valence': 1.5,      # Important for emotional content
            'energy': 1.2,       # Important for mood intensity
            'danceability': 1.0,
            'acousticness': 1.0,
            'loudness': 0.8,
            'speechiness': 0.7,
            'tempo': 0.8,
            'instrumentalness': 0.5,
            'liveness': 0.5
        }
        
        for feature, threshold in mood_thresholds.items():
            if feature in features:  # Check if feature exists
                feature_idx = features.index(feature)
                feature_value = centroid[feature_idx]
                
                # Calculate weighted difference
                difference = abs(feature_value - threshold)
                weight = feature_weights[feature]
                weighted_score += difference * weight
        return weighted_score

    cluster_moods = {}
    for i, centroid in enumerate(centroids):
        scores = {}
        for mood, thresholds in mood_mapping.items():
            score = match_mood_to_centroid(centroid, thresholds)
            scores[mood] = score
        
        # Get top 3 closest moods for this cluster
        sorted_moods = sorted(scores.items(), key=lambda x: x[1])
        best_mood = sorted_moods[0][0]
        cluster_moods[i] = best_mood
        
        # Print cluster analysis (optional)
        #print(f"Cluster {i} assigned to {best_mood} (score: {sorted_moods[0][1]:.3f})")
        #print(f"Next best matches: {sorted_moods[1][0]} ({sorted_moods[1][1]:.3f}), {sorted_moods[2][0]} ({sorted_moods[2][1]:.3f})")


    '''
    def generate_song_for_mood(mood):
        if mood not in cluster_moods.values():
            return "Mood not found"
        for k, v in cluster_moods.items():
            if v == mood:
                cluster_id = k
                break
        song_features = centroids[cluster_id] 
        song_attr = scaler.inverse_transform(song_features.reshape(1, -1))[0]  
        distances = cdist(df[features], song_attr.reshape(1, -1), metric='euclidean').flatten()
        closest_song_idx = distances[:5]
        closest_song = df.iloc[closest_song_idx].sort_values('popularity', ascending=False).head(1)
        return closest_song
    '''

    def generate_song_for_mood(mood):
        if mood not in mood_mapping:
            return "Mood not found in mapping"
            
        # Find all clusters that match this mood
        matching_clusters = [k for k, v in cluster_moods.items() if v == mood]
        
        if not matching_clusters:
            return "No clusters found for this mood"
            
        # Get the cluster that best represents this mood
        best_cluster = matching_clusters[0]
        song_features = centroids[best_cluster]
        
        # Transform features back to original scale
        song_attr = scaler.inverse_transform(song_features.reshape(1, -1))[0]
        
        # Calculate distances to all songs
        distances = cdist(df[features], song_attr.reshape(1, -1), metric='euclidean').flatten()
        
        # Get indices of 20 closest songs
        closest_indices = np.argsort(distances)[:20]
        
        # Get top 5 most popular songs from these 20
        closest_songs = df.iloc[closest_indices].sort_values('popularity', ascending=False).head(5)
        
        return closest_songs.iloc[0]  # Return the most popular matching song

    nono = {'afraid', 'content', 'faithful', 'proud', 'embarrassed', 'excited', 'ashamed', 'surprised', 'furious', 'caring', 'trusting', 'disgusted', 'anticipating', 'anxious', 'hopeful', 'impressed', 'apprehensive'}
    yesyes = mood_mapping.keys() - nono
    current_mood = max(set(predicted_labels), key=predicted_labels.count, default=predicted_labels[0])
    if current_mood in nono:
        current_mood = random.choice(list(yesyes))
    song = generate_song_for_mood(current_mood)

    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=1&q={song['track_name']}{song['artists']}&key={API_KEY}"
    response = requests.get(url)
    link = response.json()['items'][0]['id']['videoId']
    song = f"https://www.youtube.com/watch?v={link}"   
    
    result = {
        "input_text": input_text,
        "summary": summary,
        "keywords": summary_keywords,
        "latest_articles": latest_articles,
        "nearby_places": nearby_places,
        "sentences": sentences,
        "predictions": predicted_labels,
        "song": song,
    }

    return result


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  
    uvicorn.run(app, host="0.0.0.0", port=port)
