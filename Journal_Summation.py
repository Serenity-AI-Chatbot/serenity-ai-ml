import sys
import spacy
import requests
import pickle
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import defaultdict
from collections import Counter
from heapq import nlargest
import json
from googleapiclient.discovery import build
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer, util


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

'''
def get_latest_articles(keywords):
    service = build("customsearch", "v1", developerKey=API_KEY)
    query = " ".join(
        keyword
        for keyword in keywords
        if not is_common_word(keyword) and not is_date_related(keyword)
    )
    res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=5).execute()
    articles = []
    if "items" in res:
        for item in res["items"]:
            article = {
                "title": item["title"],
                "link": item["link"],
                "snippet": item["snippet"],
            }
            articles.append(article)

    return articles
'''

#embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
#embedding_model.save("Similarity_Model")

embedding_model = SentenceTransformer("Similarity_Model")

def get_latest_articles(keywords):
    service = build("customsearch", "v1", developerKey=API_KEY)

    refined_keywords = [
        keyword for keyword in keywords
        if not is_common_word(keyword) and not is_date_related(keyword)
    ]
    
    print(refined_keywords)
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
                        place = {
                            "name": result["name"],
                            "address": result["vicinity"],
                            "rating": result.get("rating", None),
                            "types": result["types"],
                            "user_ratings_total": result.get("user_ratings_total", 0)
                        }
                        places.append(place)
                        if len(places) >= limit:
                            return places
    places = sorted(places, key=lambda x: (-x["rating"] if x["rating"] else 0, -x["user_ratings_total"]))
    return places[:limit]

input_text = input("Enter the text: ")

summary = summarize_text(input_text)

summary_keywords = keywords_text(input_text)

latest_articles = get_latest_articles(summary_keywords)

location = "mumbai"
nearby_places = get_nearby_places(summary_keywords, location)

result = {
    "input_text": input_text,
    "summary": summary,
    "keywords": summary_keywords,
    "latest_articles": latest_articles,
    "nearby_places": nearby_places,
}

result_json = json.dumps(result, indent=2)
print(result_json)
