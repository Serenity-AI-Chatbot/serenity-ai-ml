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
from telegram.request import HTTPXRequest
import httpx
import asyncio
import functools

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = "https://serenity-ai-ml.onrender.com/telegram/webhook"

# Create fresh clients for each request to avoid "Event loop is closed" errors
def get_telegram_bot():
    request = HTTPXRequest(
        connection_pool_size=8,
        read_timeout=30.0,
        write_timeout=30.0,
        connect_timeout=30.0,
        pool_timeout=3.0,
    )
    return Bot(token=TELEGRAM_BOT_TOKEN, request=request)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# URL of the Node backend (adjust host/port as needed)
NODE_BACKEND_URL = os.getenv("NODE_BACKEND_URL")

# def send_message(chat_id, text):
#     # url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
#     # payload = {"chat_id": chat_id, "text": text}
#     # requests.post(url, json=payload)
#     try:
#         bot.send_message(chat_id=chat_id, text=text)
#     except Exception as e:
#         logger.error(f"Failed to send message: {e}")

@app.post("/telegram/webhook")
async def telegram_webhook(req: Request, background_tasks: BackgroundTasks):
    logger.info("Message Received")
    try:
        data = await req.json()
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
        return JSONResponse(content={"status": "error"}, status_code=200)

    # Instead of creating a background task and returning immediately,
    # we'll run the process directly and wait for it to complete
    await process_update(data)
    return JSONResponse(content={"status": "OK"}, status_code=200)

async def process_update(data):
    logger.info("Starting process_update")
    try:
        # Get a fresh bot instance for each update
        bot = get_telegram_bot()
        update = telegram.Update.de_json(data, bot)
        logger.info(update)
        
        if update.message and update.message.text:
            chat_id = update.message.chat.id
            user_message = update.message.text
            user_name = update.message.chat.username
            
            logger.info(f"Received message: '{user_message}' from user: {user_name} (ID: {chat_id})")
            
            try:
                # Log before making the API call
                logger.info(f"Calling Node backend at: {NODE_BACKEND_URL}")
                logger.info(f"Sending payload: {{'message': '{user_message}', 'from': {chat_id}, 'username': '{user_name}'}}")
                
                # Add timeout and retry logic for Vercel serverless functions
                max_retries = 3
                retry_count = 0
                last_exception = None
                
                while retry_count < max_retries:
                    try:
                        # Increased timeout to 30 seconds for cold start of Vercel functions
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            logger.info(f"Created httpx client (attempt {retry_count+1}/{max_retries})")
                            node_response = await client.post(
                                NODE_BACKEND_URL,
                                json={"message": user_message, "from": chat_id, "username": user_name}
                            )
                            logger.info(f"Node backend HTTP status: {node_response.status_code}")
                        
                        node_response.raise_for_status()
                        response_data = node_response.json()
                        logger.info(f"Node response data: {response_data}")
                        last_exception = None
                        break
                    except Exception as e:
                        last_exception = e
                        logger.warning(f"Attempt {retry_count+1} failed: {str(e)}")
                        retry_count += 1
                        await asyncio.sleep(1)  # Wait 1 second before retrying
                
                if last_exception:
                    raise last_exception
                
                bot_response = response_data.get("response", "Sorry, something went wrong.")
                logger.info(f"Bot response: {bot_response}")
                
                try:
                    logger.info(f"Sending message to chat ID: {chat_id}")
                    # Limit response length to avoid Telegram API errors
                    if len(bot_response) > 4000:
                        bot_response = bot_response[:3997] + "..."
                    await bot.send_message(chat_id=chat_id, text=bot_response)
                    logger.info("Message sent successfully")
                except telegram.error.TimedOut as e:
                    logger.error(f"Telegram API timed out: {e}")
                    # Try again with a simpler message
                    try:
                        await bot.send_message(chat_id=chat_id, text="Sorry, the response was too large or took too long. Please try again with a shorter message.")
                    except Exception as retry_e:
                        logger.error(f"Also failed to send simplified message: {retry_e}")
                except Exception as e:
                    logger.error(f"Failed to send message: {e}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")

            except Exception as e:
                logger.error(f"Error calling Node backend: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                try:
                    await bot.send_message(chat_id, text="Sorry, something went wrong connecting to our services.")
                except Exception as send_err:
                    logger.error(f"Also failed to send error message: {send_err}")
    except Exception as e:
        logger.error(f"General error in process_update: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    # Use UVLoop if available for better performance
    try:
        import uvloop
        uvloop.install()
        logger.info("Using uvloop for improved performance")
    except ImportError:
        logger.info("uvloop not available, using default asyncio event loop")
    
    # Clear any existing event loop policies
    asyncio.set_event_loop_policy(None)
    
    # Create a new event loop for this instance
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    port = int(os.environ.get('PORT', 5000))  
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
