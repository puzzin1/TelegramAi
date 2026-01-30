#!/usr/bin/env python3
"""
Telegram image-to-OpenAI bot with detailed logging and improved UI
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
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackQueryHandler, filters

# ---------- Configuration ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_TELEGRAM_ID = os.getenv("ADMIN_TELEGRAM_ID")
MODEL = os.getenv("MODEL", "gpt-4o-mini")
DB_PATH = os.getenv("BOT_DB", "bot_users.db")
# -----------------------------------

# Enhanced logging configuration
logging.basicConfig(
    level=logging.DEBUG,
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
                
                import json
                j = json.loads(text)
                logger.debug(f"Response JSON keys: {j.keys()}")
                
                if "choices" in j and len(j["choices"]) > 0:
                    message_content = j["choices"][0].get("message", {}).get("content")
                    if isinstance(message_content, str):
                        logger.info(f"OpenAI response length: {len(message_content)} chars")
                        logger.info("OPENAI REQUEST END")
                        logger.info("=" * 60)
                        return message_content
                
                logger.warning("Unexpected response structure from OpenAI")
                return str(j)
                
    except asyncio.TimeoutError:
        logger.error("OpenAI request timeout")
        return "Timeout: OpenAI не ответил вовремя."
    except Exception as e:
        logger.error(f"OpenAI request failed: {e}")
        logger.error(traceback.format_exc())
        return f"Ошибка при обращении к OpenAI: {e}"


# --- Menu keyboards ---
def get_main_menu_keyboard(is_admin: bool = False):
    """Create main menu keyboard with buttons"""
    keyboard = [
        [KeyboardButton("📱 Мой Telegram ID")],
    ]
    
    if is_admin:
        keyboard.extend([
            [KeyboardButton("👥 Список пользователей")],
            [KeyboardButton("➕ Добавить пользователя"), KeyboardButton("➖ Удалить пользователя")],
        ])
    
    keyboard.append([KeyboardButton("ℹ️ Помощь")])
    
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


def get_inline_menu_keyboard(is_admin: bool = False):
    """Create inline keyboard menu"""
    buttons = [
        [InlineKeyboardButton("📱 Мой Telegram ID", callback_data="my_id")],
    ]
    
    if is_admin:
        buttons.extend([
            [InlineKeyboardButton("👥 Список пользователей", callback_data="list_users")],
            [InlineKeyboardButton("ℹ️ Помощь", callback_data="help")],
        ])
    else:
        buttons.append([InlineKeyboardButton("ℹ️ Помощь", callback_data="help")])
    
    return InlineKeyboardMarkup(buttons)


# --- Command handlers ---
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("=" * 60)
    logger.info("/start command received")
    uid = update.effective_user.id
    username = update.effective_user.username
    logger.info(f"User: {uid} (@{username})")
    
    is_admin = (uid == ADMIN_TELEGRAM_ID)
    is_allowed = db.is_allowed(uid) or is_admin
    
    welcome_text = f"""
🤖 <b>Добро пожаловать в AI Image Bot!</b>

👋 Привет, {update.effective_user.first_name}!

<b>Что я умею:</b>
🖼 Анализировать изображения
💬 Отвечать на текстовые вопросы
🧠 Использую модель: {MODEL}

<b>Как пользоваться:</b>
1️⃣ Отправьте мне изображение с подписью-вопросом
2️⃣ Или просто напишите текстовое сообщение

"""
    
    if is_admin:
        welcome_text += """
<b>👑 Вы администратор бота!</b>

<b>Доступные команды:</b>
/add <code>telegram_id</code> - добавить пользователя
/remove <code>telegram_id</code> - удалить пользователя
/list - список всех пользователей
/menu - показать меню

<b>Используйте кнопки ниже для быстрого доступа:</b>
"""
    elif is_allowed:
        welcome_text += "✅ <b>У вас есть доступ к боту!</b>\n\n"
    else:
        welcome_text += "❌ <b>У вас пока нет доступа к боту.</b>\nОбратитесь к администратору.\n\n"
    
    welcome_text += "\n📱 <b>Используйте меню ниже для навигации</b>"
    
    keyboard = get_main_menu_keyboard(is_admin=is_admin)
    
    await update.message.reply_text(
        welcome_text,
        parse_mode='HTML',
        reply_markup=keyboard
    )
    
    logger.info("Welcome message sent")
    logger.info("=" * 60)


async def menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show menu with inline buttons"""
    logger.info("/menu command received")
    uid = update.effective_user.id
    is_admin = (uid == ADMIN_TELEGRAM_ID)
    
    menu_text = """
📋 <b>Главное меню</b>

Выберите действие из списка ниже:
"""
    
    keyboard = get_inline_menu_keyboard(is_admin=is_admin)
    
    await update.message.reply_text(
        menu_text,
        parse_mode='HTML',
        reply_markup=keyboard
    )


async def my_id_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show user's Telegram ID - available for everyone"""
    logger.info("My ID request")
    uid = update.effective_user.id
    username = update.effective_user.username or "не установлен"
    first_name = update.effective_user.first_name or ""
    last_name = update.effective_user.last_name or ""
    
    full_name = f"{first_name} {last_name}".strip() or "не указано"
    
    id_text = f"""
📱 <b>Ваша информация:</b>

🆔 <b>Telegram ID:</b> <code>{uid}</code>
👤 <b>Имя:</b> {full_name}
📝 <b>Username:</b> @{username}

<i>Нажмите на ID, чтобы скопировать</i>
"""
    
    logger.info(f"Sent ID info to user {uid}")
    
    await update.message.reply_text(id_text, parse_mode='HTML')


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show help information"""
    logger.info("Help request")
    uid = update.effective_user.id
    is_admin = (uid == ADMIN_TELEGRAM_ID)
    
    help_text = """
ℹ️ <b>Справка по использованию бота</b>

<b>🖼 Работа с изображениями:</b>
• Отправьте изображение с подписью-вопросом
• Бот проанализирует изображение и ответит на ваш вопрос
• Примеры: "Что на этой картинке?", "Опиши подробно", "Найди текст"

<b>💬 Текстовые запросы:</b>
• Просто напишите сообщение боту
• Бот ответит используя AI модель {MODEL}

<b>📱 Основные команды:</b>
/start - перезапустить бота
/menu - показать меню
/myid - узнать свой Telegram ID
/help - эта справка

"""
    
    if is_admin:
        help_text += """
<b>👑 Команды администратора:</b>
/add <code>telegram_id</code> - добавить пользователя
/remove <code>telegram_id</code> - удалить пользователя
/list - список всех пользователей

<b>Примеры:</b>
<code>/add 123456789</code>
<code>/remove 123456789</code>
"""
    
    help_text += "\n💡 <b>Совет:</b> Для лучших результатов формулируйте вопросы четко и конкретно!"
    
    await update.message.reply_text(help_text, parse_mode='HTML')


async def add_user_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("=" * 60)
    logger.info("/add command received")
    uid = update.effective_user.id
    logger.info(f"From user: {uid}")
    
    if uid != ADMIN_TELEGRAM_ID:
        logger.warning(f"Unauthorized add attempt by {uid}")
        await update.message.reply_text("❌ Эта команда доступна только администратору.")
        return
    
    if not context.args or len(context.args) != 1:
        await update.message.reply_text(
            "⚠️ <b>Неверный формат команды!</b>\n\n"
            "Используйте: <code>/add telegram_id</code>\n"
            "Пример: <code>/add 123456789</code>",
            parse_mode='HTML'
        )
        return
    
    try:
        target_id = int(context.args[0])
        logger.info(f"Adding user: {target_id}")
        db.add_user(target_id, None)
        await update.message.reply_text(
            f"✅ <b>Пользователь добавлен!</b>\n\n"
            f"🆔 Telegram ID: <code>{target_id}</code>",
            parse_mode='HTML'
        )
        logger.info(f"User {target_id} added successfully")
    except ValueError as e:
        logger.error(f"Invalid telegram_id format: {e}")
        await update.message.reply_text(
            "❌ <b>Ошибка!</b>\n\n"
            "Telegram ID должен быть числом.\n"
            "Пример: <code>/add 123456789</code>",
            parse_mode='HTML'
        )
    except Exception as e:
        logger.error(f"Add user failed: {e}")
        await update.message.reply_text(f"❌ Ошибка при добавлении пользователя: {e}")
    
    logger.info("=" * 60)


async def remove_user_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("=" * 60)
    logger.info("/remove command received")
    uid = update.effective_user.id
    logger.info(f"From user: {uid}")
    
    if uid != ADMIN_TELEGRAM_ID:
        logger.warning(f"Unauthorized remove attempt by {uid}")
        await update.message.reply_text("❌ Эта команда доступна только администратору.")
        return
    
    if not context.args or len(context.args) != 1:
        await update.message.reply_text(
            "⚠️ <b>Неверный формат команды!</b>\n\n"
            "Используйте: <code>/remove telegram_id</code>\n"
            "Пример: <code>/remove 123456789</code>",
            parse_mode='HTML'
        )
        return
    
    try:
        target_id = int(context.args[0])
        logger.info(f"Removing user: {target_id}")
        db.remove_user(target_id)
        await update.message.reply_text(
            f"✅ <b>Пользователь удалён!</b>\n\n"
            f"🆔 Telegram ID: <code>{target_id}</code>",
            parse_mode='HTML'
        )
        logger.info(f"User {target_id} removed successfully")
    except ValueError as e:
        logger.error(f"Invalid telegram_id format: {e}")
        await update.message.reply_text(
            "❌ <b>Ошибка!</b>\n\n"
            "Telegram ID должен быть числом.\n"
            "Пример: <code>/remove 123456789</code>",
            parse_mode='HTML'
        )
    except Exception as e:
        logger.error(f"Remove user failed: {e}")
        await update.message.reply_text(f"❌ Ошибка при удалении пользователя: {e}")
    
    logger.info("=" * 60)


async def list_users_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("=" * 60)
    logger.info("/list command received")
    uid = update.effective_user.id
    logger.info(f"From user: {uid}")
    
    if uid != ADMIN_TELEGRAM_ID:
        logger.warning(f"Unauthorized list attempt by {uid}")
        await update.message.reply_text("❌ Эта команда доступна только администратору.")
        return
    
    rows = db.list_users()
    
    if not rows:
        await update.message.reply_text("📭 <b>Список пользователей пуст.</b>", parse_mode='HTML')
        return
    
    text = "👥 <b>Список разрешённых пользователей:</b>\n\n"
    
    for i, (tid, username, added_at) in enumerate(rows, 1):
        username_display = f"@{username}" if username else "не указан"
        text += f"{i}. 🆔 <code>{tid}</code>\n"
        text += f"   👤 {username_display}\n"
        text += f"   📅 Добавлен: {added_at}\n\n"
        logger.debug(f"User: {tid}, {username}, {added_at}")
    
    text += f"<b>Всего пользователей:</b> {len(rows)}"
    
    await update.message.reply_text(text, parse_mode='HTML')
    logger.info("=" * 60)


# --- Callback query handler for inline buttons ---
async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard button presses"""
    query = update.callback_query
    await query.answer()
    
    uid = query.from_user.id
    is_admin = (uid == ADMIN_TELEGRAM_ID)
    
    callback_data = query.data
    logger.info(f"Button callback: {callback_data} from user {uid}")
    
    if callback_data == "my_id":
        username = query.from_user.username or "не установлен"
        first_name = query.from_user.first_name or ""
        last_name = query.from_user.last_name or ""
        full_name = f"{first_name} {last_name}".strip() or "не указано"
        
        id_text = f"""
📱 <b>Ваша информация:</b>

🆔 <b>Telegram ID:</b> <code>{uid}</code>
👤 <b>Имя:</b> {full_name}
📝 <b>Username:</b> @{username}

<i>Нажмите на ID, чтобы скопировать</i>
"""
        await query.edit_message_text(id_text, parse_mode='HTML')
        
    elif callback_data == "list_users" and is_admin:
        rows = db.list_users()
        
        if not rows:
            await query.edit_message_text("📭 <b>Список пользователей пуст.</b>", parse_mode='HTML')
            return
        
        text = "👥 <b>Список разрешённых пользователей:</b>\n\n"
        
        for i, (tid, username, added_at) in enumerate(rows, 1):
            username_display = f"@{username}" if username else "не указан"
            text += f"{i}. 🆔 <code>{tid}</code> ({username_display})\n"
        
        text += f"\n<b>Всего:</b> {len(rows)}"
        await query.edit_message_text(text, parse_mode='HTML')
        
    elif callback_data == "help":
        help_text = """
ℹ️ <b>Справка по использованию бота</b>

<b>🖼 Работа с изображениями:</b>
• Отправьте изображение с подписью-вопросом
• Бот проанализирует и ответит

<b>💬 Текстовые запросы:</b>
• Напишите сообщение боту
• Получите ответ от AI

<b>📱 Команды:</b>
/start - перезапуск
/menu - меню
/myid - ваш ID
/help - справка

💡 <b>Совет:</b> Формулируйте вопросы четко!
"""
        await query.edit_message_text(help_text, parse_mode='HTML')


# --- Message handlers for keyboard buttons ---
async def keyboard_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle keyboard button presses"""
    text = update.message.text
    uid = update.effective_user.id
    is_admin = (uid == ADMIN_TELEGRAM_ID)
    
    logger.info(f"Keyboard button pressed: {text} by user {uid}")
    
    if text == "📱 Мой Telegram ID":
        await my_id_handler(update, context)
    elif text == "ℹ️ Помощь":
        await help_handler(update, context)
    elif text == "👥 Список пользователей" and is_admin:
        await list_users_handler(update, context)
    elif text == "➕ Добавить пользователя" and is_admin:
        await update.message.reply_text(
            "➕ <b>Добавление пользователя</b>\n\n"
            "Используйте команду:\n"
            "<code>/add telegram_id</code>\n\n"
            "Пример: <code>/add 123456789</code>",
            parse_mode='HTML'
        )
    elif text == "➖ Удалить пользователя" and is_admin:
        await update.message.reply_text(
            "➖ <b>Удаление пользователя</b>\n\n"
            "Используйте команду:\n"
            "<code>/remove telegram_id</code>\n\n"
            "Пример: <code>/remove 123456789</code>",
            parse_mode='HTML'
        )


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("=" * 60)
    logger.info("PHOTO RECEIVED")
    uid = update.effective_user.id
    username = update.effective_user.username
    logger.info(f"From user: {uid} (@{username})")
    
    if not db.is_allowed(uid) and uid != ADMIN_TELEGRAM_ID:
        logger.warning(f"Unauthorized user {uid} tried to send photo")
        await update.message.reply_text(
            "❌ <b>Вы не авторизованы для использования бота.</b>\n\n"
            "Обратитесь к администратору для получения доступа.",
            parse_mode='HTML'
        )
        return
    
    caption = update.message.caption or "Опишите изображение."
    logger.info(f"Caption: {caption}")
    
    await update.message.reply_text("🔄 Принял изображение, обрабатываю...")
    
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
        await update.message.reply_text(f"✅ <b>Результат анализа:</b>\n\n{resp_text}", parse_mode='HTML')
        logger.info("Response sent successfully")
        
        # cleanup
        logger.debug(f"Cleaning up temp file: {p}")
        p.unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"Photo handler failed: {e}")
        logger.error(traceback.format_exc())
        await update.message.reply_text(f"❌ <b>Ошибка при обработке изображения:</b>\n\n{e}", parse_mode='HTML')
    
    logger.info("=" * 60)


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("=" * 60)
    logger.info("TEXT MESSAGE RECEIVED")
    uid = update.effective_user.id
    username = update.effective_user.username
    logger.info(f"From user: {uid} (@{username})")
    
    if not db.is_allowed(uid) and uid != ADMIN_TELEGRAM_ID:
        logger.warning(f"Unauthorized user {uid} tried to send text")
        await update.message.reply_text(
            "❌ <b>Вы не авторизованы для использования бота.</b>\n\n"
            "Обратитесь к администратору для получения доступа.",
            parse_mode='HTML'
        )
        return
    
    user_text = update.message.text
    logger.info(f"Text: {user_text[:100]}...")
    
    await update.message.reply_text("🔄 Запрос отправлен в OpenAI, ждите...")
    
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
                        await update.message.reply_text(f"✅ <b>Ответ:</b>\n\n{content}", parse_mode='HTML')
                        logger.info("Response sent successfully")
                        logger.info("=" * 60)
                        return
                
                logger.warning("Unexpected response format")
                await update.message.reply_text(str(j))
                
    except Exception as e:
        logger.error(f"Text handler failed: {e}")
        logger.error(traceback.format_exc())
        txt = str(e)
        await update.message.reply_text(f"❌ <b>Ошибка от OpenAI:</b>\n\n{txt}", parse_mode='HTML')
    
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
        
        app.add_handler(CommandHandler("menu", menu_handler))
        logger.info("Handler registered: /menu")
        
        app.add_handler(CommandHandler("myid", my_id_handler))
        logger.info("Handler registered: /myid")
        
        app.add_handler(CommandHandler("help", help_handler))
        logger.info("Handler registered: /help")
        
        app.add_handler(CommandHandler("add", add_user_handler))
        logger.info("Handler registered: /add")
        
        app.add_handler(CommandHandler("remove", remove_user_handler))
        logger.info("Handler registered: /remove")
        
        app.add_handler(CommandHandler("list", list_users_handler))
        logger.info("Handler registered: /list")
        
        # Add callback query handler for inline buttons
        app.add_handler(CallbackQueryHandler(button_callback_handler))
        logger.info("Handler registered: CALLBACK_QUERY")
        
        # Add keyboard button handler (must be before general text handler)
        app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND & 
            (filters.Regex("^📱 Мой Telegram ID$") | 
             filters.Regex("^ℹ️ Помощь$") | 
             filters.Regex("^👥 Список пользователей$") |
             filters.Regex("^➕ Добавить пользователя$") |
             filters.Regex("^➖ Удалить пользователя$")),
            keyboard_button_handler
        ))
        logger.info("Handler registered: KEYBOARD_BUTTONS")
        
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