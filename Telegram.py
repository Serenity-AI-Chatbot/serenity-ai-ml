from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import requests
import json
import requests
from dotenv import load_dotenv
import os
from loguru import logger 
from telegram import Bot
import telegram
import httpx

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = "https://serenity-ai-ml.onrender.com/telegram/webhook"

bot = Bot(token=TELEGRAM_BOT_TOKEN)

app = FastAPI()

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
        user_name = update.message.chat.username
        
        try:
            # here i am calling our node backend for gemini ka response :)
            node_response = requests.post(
                NODE_BACKEND_URL,
                json={"message": user_message, "from": chat_id, 'username': user_name}
            )
            node_response.raise_for_status()
            response_data = node_response.json()
            bot_response = response_data.get("response", "Sorry, something went wrong.")
            send_message(chat_id, bot_response)

        except Exception as e:
            logger.error(f"Error calling Node backend: {e}")
            send_message(chat_id, "Sorry, something went wrong.")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  
    uvicorn.run(app, host="0.0.0.0", port=port)
