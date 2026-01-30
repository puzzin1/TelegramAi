#!/usr/bin/env python3
"""
Telegram image-to-OpenAI bot with detailed logging
"""

import os
import asyncio
import logging
import base64
import sqlite3
from pathlib import Path
from typing import Optional
import traceback

import aiohttp
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# ---------- Configuration ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_TELEGRAM_ID = os.getenv("ADMIN_TELEGRAM_ID")
MODEL = os.getenv("MODEL", "gpt-4o-mini")
DB_PATH = os.getenv("BOT_DB", "bot_users.db")
# -----------------------------------

# Enhanced logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more details
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('telegram_bot_debug.log')
    ]
)
logger = logging.getLogger(__name__)

def _mask(s):
    if not s:
        return '<MISSING>'
    s = str(s)
    if len(s) <= 6:
        return '***'
    return s[:3] + '...' + s[-3:]

# Log startup configuration
logger.info("=" * 80)
logger.info("BOT STARTUP - Configuration check")
logger.info("=" * 80)
logger.info(f"TELEGRAM_TOKEN: {_mask(TELEGRAM_TOKEN)}")
logger.info(f"OPENAI_API_KEY: {_mask(OPENAI_API_KEY)}")
logger.info(f"ADMIN_TELEGRAM_ID: {ADMIN_TELEGRAM_ID}")
logger.info(f"MODEL: {MODEL}")
logger.info(f"DB_PATH: {DB_PATH}")
logger.info("=" * 80)

if not TELEGRAM_TOKEN or not OPENAI_API_KEY or not ADMIN_TELEGRAM_ID:
    logger.error("FATAL: Missing required environment variables!")
    logger.error(f"TELEGRAM_TOKEN present: {bool(TELEGRAM_TOKEN)}")
    logger.error(f"OPENAI_API_KEY present: {bool(OPENAI_API_KEY)}")
    logger.error(f"ADMIN_TELEGRAM_ID present: {bool(ADMIN_TELEGRAM_ID)}")
    raise SystemExit("Please set TELEGRAM_TOKEN, OPENAI_API_KEY and ADMIN_TELEGRAM_ID environment variables.")

try:
    ADMIN_TELEGRAM_ID = int(ADMIN_TELEGRAM_ID)
    logger.info(f"Admin ID parsed successfully: {ADMIN_TELEGRAM_ID}")
except ValueError as e:
    logger.error(f"FATAL: Cannot parse ADMIN_TELEGRAM_ID as integer: {e}")
    raise


# --- Simple SQLite wrapper ---
class DB:
    def __init__(self, path: str = DB_PATH):
        logger.info(f"Initializing database at: {path}")
        self.path = path
        try:
            self._init_db()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def _init_db(self):
        logger.debug(f"Creating database tables if not exist...")
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            telegram_id INTEGER PRIMARY KEY,
            username TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id INTEGER,
            action TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        conn.close()
        logger.debug("Database tables ready")

    def add_user(self, telegram_id: int, username: Optional[str]):
        logger.info(f"Adding user: telegram_id={telegram_id}, username={username}")
        try:
            conn = sqlite3.connect(self.path)
            cur = conn.cursor()
            cur.execute("INSERT OR IGNORE INTO users(telegram_id, username) VALUES(?,?)", (telegram_id, username))
            cur.execute("INSERT INTO logs(telegram_id, action) VALUES(?,?)", (telegram_id, 'add'))
            conn.commit()
            conn.close()
            logger.info(f"User {telegram_id} added successfully")
        except Exception as e:
            logger.error(f"Failed to add user {telegram_id}: {e}")
            raise

    def remove_user(self, telegram_id: int):
        logger.info(f"Removing user: telegram_id={telegram_id}")
        try:
            conn = sqlite3.connect(self.path)
            cur = conn.cursor()
            cur.execute("DELETE FROM users WHERE telegram_id = ?", (telegram_id,))
            cur.execute("INSERT INTO logs(telegram_id, action) VALUES(?,?)", (telegram_id, 'remove'))
            conn.commit()
            conn.close()
            logger.info(f"User {telegram_id} removed successfully")
        except Exception as e:
            logger.error(f"Failed to remove user {telegram_id}: {e}")
            raise

    def is_allowed(self, telegram_id: int) -> bool:
        logger.debug(f"Checking if user {telegram_id} is allowed")
        try:
            conn = sqlite3.connect(self.path)
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM users WHERE telegram_id = ?", (telegram_id,))
            r = cur.fetchone()
            conn.close()
            result = bool(r)
            logger.debug(f"User {telegram_id} allowed: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to check user {telegram_id}: {e}")
            return False

    def list_users(self):
        logger.debug("Listing all users")
        try:
            conn = sqlite3.connect(self.path)
            cur = conn.cursor()
            cur.execute("SELECT telegram_id, username, added_at FROM users ORDER BY added_at DESC")
            rows = cur.fetchall()
            conn.close()
            logger.debug(f"Found {len(rows)} users")
            return rows
        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            return []

# instantiate DB
logger.info("Creating database instance...")
db = DB()

# --- OpenAI call helper ---
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

async def query_openai_with_image(b64_data: str, prompt: str) -> str:
    """Send image + prompt to OpenAI and return text response"""
    logger.info("=" * 60)
    logger.info("OPENAI REQUEST START")
    logger.info(f"Prompt: {prompt[:100]}...")
    logger.info(f"Image data length: {len(b64_data)} bytes")
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    
    data_uri = "data:image/png;base64," + b64_data
    logger.debug(f"Data URI length: {len(data_uri)}")
    
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }
        ],
        "max_completion_tokens": 10000,
        "reasoning_effort": "high",
        "temperature": 1
    }
    
    logger.debug(f"Payload structure: model={MODEL}, messages count={len(payload['messages'])}")
    
    try:
        async with aiohttp.ClientSession() as session:
            logger.info(f"Sending POST request to {OPENAI_CHAT_URL}")
            async with session.post(OPENAI_CHAT_URL, json=payload, headers=headers, timeout=240) as resp:
                status = resp.status
                logger.info(f"OpenAI response status: {status}")
                
                text = await resp.text()
                logger.debug(f"Response text length: {len(text)}")
                
                if status != 200:
                    logger.error(f"OpenAI error response: {text}")
                    return f"Ошибка от OpenAI: {status}. Смотрите логи. Ответ: {text[:200]}"
                
                try:
                    j = await resp.json()
                    logger.debug(f"Response JSON keys: {j.keys()}")
                    
                    if "choices" in j and len(j["choices"]) > 0:
                        msg = j["choices"][0].get("message", {})
                        content = msg.get("content")
                        logger.info(f"Content type: {type(content)}")
                        
                        if isinstance(content, str):
                            logger.info(f"Response content length: {len(content)} chars")
                            logger.info("OPENAI REQUEST SUCCESS")
                            logger.info("=" * 60)
                            return content
                        
                        if isinstance(content, list):
                            logger.debug("Content is list, parsing...")
                            parts = []
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    parts.append(item.get("text", ""))
                                elif isinstance(item, str):
                                    parts.append(item)
                            result = "\n".join(p for p in parts if p)
                            logger.info(f"Parsed {len(parts)} content parts")
                            logger.info("OPENAI REQUEST SUCCESS")
                            logger.info("=" * 60)
                            return result
                    
                    logger.warning("Unexpected response structure, returning raw JSON")
                    return str(j)
                    
                except Exception as e:
                    logger.error(f"Failed to parse OpenAI JSON: {e}")
                    logger.error(traceback.format_exc())
                    return "Не удалось разобрать ответ OpenAI."
                    
    except asyncio.TimeoutError:
        logger.error("OpenAI request timeout (240s)")
        return "Таймаут запроса к OpenAI (240 секунд)"
    except Exception as e:
        logger.error(f"OpenAI request failed: {e}")
        logger.error(traceback.format_exc())
        raise


# --- Telegram handlers ---
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("=" * 60)
    logger.info("COMMAND: /start")
    uid = update.effective_user.id
    username = update.effective_user.username
    logger.info(f"User: {uid} (@{username})")
    
    if uid == ADMIN_TELEGRAM_ID:
        logger.info("User is admin")
        await update.message.reply_text("Привет! Я бот. Вы — админ. Используйте /add, /remove, /list.")
    else:
        allowed = db.is_allowed(uid)
        logger.info(f"User allowed: {allowed}")
        if allowed:
            await update.message.reply_text("Привет! Отправь картинку и вопрос — я перешлю её в OpenAI.")
        else:
            await update.message.reply_text("Вы не авторизованы для использования бота. Обратитесь к администратору.")
    logger.info("=" * 60)

async def add_user_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("=" * 60)
    logger.info("COMMAND: /add")
    logger.info(f"Caller: {update.effective_user.id}")
    logger.info(f"Args: {context.args}")
    
    if update.effective_user.id != ADMIN_TELEGRAM_ID:
        logger.warning("Non-admin tried to add user")
        await update.message.reply_text("Только админ может добавлять пользователей.")
        return
    
    if len(context.args) < 1:
        logger.warning("Missing telegram_id argument")
        await update.message.reply_text("Использование: /add <telegram_id> [username]")
        return
    
    try:
        tid = int(context.args[0])
        username = context.args[1] if len(context.args) >= 2 else None
        logger.info(f"Adding user: tid={tid}, username={username}")
        db.add_user(tid, username)
        await update.message.reply_text(f"Пользователь {tid} добавлен.")
    except ValueError as e:
        logger.error(f"Invalid telegram_id: {context.args[0]}")
        await update.message.reply_text("telegram_id должен быть числом")
    except Exception as e:
        logger.error(f"Failed to add user: {e}")
        logger.error(traceback.format_exc())
        await update.message.reply_text(f"Ошибка при добавлении: {e}")
    
    logger.info("=" * 60)

async def remove_user_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("=" * 60)
    logger.info("COMMAND: /remove")
    logger.info(f"Caller: {update.effective_user.id}")
    logger.info(f"Args: {context.args}")
    
    if update.effective_user.id != ADMIN_TELEGRAM_ID:
        logger.warning("Non-admin tried to remove user")
        await update.message.reply_text("Только админ может удалять пользователей.")
        return
    
    if len(context.args) < 1:
        logger.warning("Missing telegram_id argument")
        await update.message.reply_text("Использование: /remove <telegram_id>")
        return
    
    try:
        tid = int(context.args[0])
        logger.info(f"Removing user: tid={tid}")
        db.remove_user(tid)
        await update.message.reply_text(f"Пользователь {tid} удалён.")
    except ValueError:
        logger.error(f"Invalid telegram_id: {context.args[0]}")
        await update.message.reply_text("telegram_id должен быть числом")
    except Exception as e:
        logger.error(f"Failed to remove user: {e}")
        logger.error(traceback.format_exc())
        await update.message.reply_text(f"Ошибка при удалении: {e}")
    
    logger.info("=" * 60)

async def list_users_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("=" * 60)
    logger.info("COMMAND: /list")
    logger.info(f"Caller: {update.effective_user.id}")
    
    if update.effective_user.id != ADMIN_TELEGRAM_ID:
        logger.warning("Non-admin tried to list users")
        await update.message.reply_text("Только админ может просматривать список пользователей.")
        return
    
    rows = db.list_users()
    logger.info(f"Users count: {len(rows)}")
    
    if not rows:
        await update.message.reply_text("Список пользователей пуст.")
        return
    
    text = "Список разрешённых пользователей:\n"
    for tid, username, added_at in rows:
        text += f"- {tid} ({username}) added {added_at}\n"
        logger.debug(f"User: {tid}, {username}, {added_at}")
    
    await update.message.reply_text(text)
    logger.info("=" * 60)

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("=" * 60)
    logger.info("PHOTO RECEIVED")
    uid = update.effective_user.id
    username = update.effective_user.username
    logger.info(f"From user: {uid} (@{username})")
    
    if not db.is_allowed(uid) and uid != ADMIN_TELEGRAM_ID:
        logger.warning(f"Unauthorized user {uid} tried to send photo")
        await update.message.reply_text("Вы не авторизованы для использования бота.")
        return
    
    caption = update.message.caption or "Опишите изображение."
    logger.info(f"Caption: {caption}")
    
    await update.message.reply_text("Принял изображение, обрабатываю...")
    
    try:
        # get highest-resolution photo
        photo = update.message.photo[-1]
        logger.info(f"Photo file_id: {photo.file_id}")
        logger.info(f"Photo file_unique_id: {photo.file_unique_id}")
        logger.info(f"Photo dimensions: {photo.width}x{photo.height}")
        
        logger.debug("Getting file from Telegram...")
        file = await context.bot.get_file(photo.file_id)
        logger.info(f"File path: {file.file_path}")
        
        tmpdir = Path("/tmp/telegram_bot_images")
        tmpdir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Temp directory: {tmpdir}")
        
        p = tmpdir / f"img_{photo.file_unique_id}.jpg"
        logger.info(f"Downloading to: {p}")
        
        await file.download_to_drive(custom_path=str(p))
        logger.info(f"Download complete. File size: {p.stat().st_size} bytes")
        
        # read and base64 encode
        logger.debug("Reading and encoding image...")
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        logger.info(f"Base64 encoded, length: {len(b64)} chars")
        
        # query OpenAI
        logger.info("Querying OpenAI...")
        resp_text = await query_openai_with_image(b64, caption)
        logger.info(f"OpenAI response received, length: {len(resp_text)} chars")
        
        # reply to user
        logger.info("Sending response to user...")
        await update.message.reply_text(resp_text)
        logger.info("Response sent successfully")
        
        # cleanup
        logger.debug(f"Cleaning up temp file: {p}")
        p.unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"Photo handler failed: {e}")
        logger.error(traceback.format_exc())
        await update.message.reply_text(f"Ошибка при обработке изображения: {e}")
    
    logger.info("=" * 60)

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("=" * 60)
    logger.info("TEXT MESSAGE RECEIVED")
    uid = update.effective_user.id
    username = update.effective_user.username
    logger.info(f"From user: {uid} (@{username})")
    
    if not db.is_allowed(uid) and uid != ADMIN_TELEGRAM_ID:
        logger.warning(f"Unauthorized user {uid} tried to send text")
        await update.message.reply_text("Вы не авторизованы для использования бота.")
        return
    
    user_text = update.message.text
    logger.info(f"Text: {user_text[:100]}...")
    
    await update.message.reply_text("Запрос отправлен в OpenAI, ждите...")
    
    try:
        logger.info("Sending text-only request to OpenAI...")
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": user_text}],
                "reasoning_effort": "high",
                "max_completion_tokens": 10000
            }
            
            logger.debug(f"Payload: model={MODEL}, text_length={len(user_text)}")
            
            async with session.post(OPENAI_CHAT_URL, json=payload, headers=headers, timeout=240) as resp:
                logger.info(f"OpenAI response status: {resp.status}")
                
                j = await resp.json()
                logger.debug(f"Response keys: {j.keys()}")
                
                if "choices" in j and len(j["choices"]) > 0:
                    content = j["choices"][0].get("message", {}).get("content")
                    if isinstance(content, str):
                        logger.info(f"Response length: {len(content)} chars")
                        await update.message.reply_text(content)
                        logger.info("Response sent successfully")
                        logger.info("=" * 60)
                        return
                
                logger.warning("Unexpected response format")
                await update.message.reply_text(str(j))
                
    except Exception as e:
        logger.error(f"Text handler failed: {e}")
        logger.error(traceback.format_exc())
        txt = str(e)
        await update.message.reply_text(f"Ошибка от OpenAI: {txt}")
    
    logger.info("=" * 60)

# --- Error handler ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("=" * 60)
    logger.error("UNHANDLED ERROR")
    logger.error(f"Update: {update}")
    logger.error(f"Error: {context.error}")
    logger.error(traceback.format_exc())
    logger.error("=" * 60)

# --- App startup ---
async def main():
    logger.info("=" * 80)
    logger.info("BUILDING TELEGRAM APPLICATION")
    logger.info("=" * 80)
    
    try:
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        logger.info("Application built successfully")
        
        # Add error handler
        app.add_error_handler(error_handler)
        logger.info("Error handler registered")
        
        # Add command handlers
        app.add_handler(CommandHandler("start", start_handler))
        logger.info("Handler registered: /start")
        
        app.add_handler(CommandHandler("add", add_user_handler))
        logger.info("Handler registered: /add")
        
        app.add_handler(CommandHandler("remove", remove_user_handler))
        logger.info("Handler registered: /remove")
        
        app.add_handler(CommandHandler("list", list_users_handler))
        logger.info("Handler registered: /list")
        
        app.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, photo_handler))
        logger.info("Handler registered: PHOTO")
        
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
        logger.info("Handler registered: TEXT")
        
        logger.info("=" * 80)
        logger.info("STARTING BOT")
        logger.info("=" * 80)
        
        await app.initialize()
        logger.info("App initialized")
        
        await app.start()
        logger.info("App started")
        
        await app.updater.start_polling()
        logger.info("Polling started - BOT IS NOW RUNNING")
        logger.info("=" * 80)
        
        # run until cancelled
        try:
            await asyncio.Event().wait()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutdown signal received")
        finally:
            logger.info("Stopping bot...")
            await app.updater.stop()
            logger.info("Updater stopped")
            await app.stop()
            logger.info("App stopped")
            await app.shutdown()
            logger.info("App shutdown complete")
            
    except Exception as e:
        logger.error(f"FATAL ERROR during startup: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"FATAL: {e}")
        logger.error(traceback.format_exc())
        raise