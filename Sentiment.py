from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.tokenize import sent_tokenize
import uvicorn

nltk.download("punkt")

app = FastAPI()

label_mapping = {'sentimental': 0, 'afraid': 1, 'proud': 2, 'faithful': 3, 'terrified': 4, 'joyful': 5, 'angry': 6, 'sad': 7, 'jealous': 8, 'grateful': 9, 'prepared': 10, 'embarrassed': 11, 'excited': 12, 'annoyed': 13, 'lonely': 14, 'ashamed': 15, 'guilty': 16, 'surprised': 17, 'nostalgic': 18, 'confident': 19, 'furious': 20, 'disappointed': 21, 'caring': 22, 'trusting': 23, 'disgusted': 24, 'anticipating': 25, 'anxious': 26, 'hopeful': 27, 'content': 28, 'impressed': 29, 'apprehensive': 30, 'devastated': 31}

model_name_or_path = "Sentiment_Model"  
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

reverse_label_mapping = {v: k for k, v in label_mapping.items()}

class SentimentRequest(BaseModel):
    text: str 

@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    sentences = sent_tokenize(request.text)
    
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    
    predicted_labels = [reverse_label_mapping[label_id] for label_id in predictions.tolist()]
    
    return {"sentences": sentences, "predictions": predicted_labels}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
