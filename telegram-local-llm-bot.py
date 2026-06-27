from dotenv import load_dotenv
load_dotenv()

import os
import random
import base64
import requests
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

# Словарь для хранения истории диалога
conversation_history = {}

# Получаем токен из переменной окружения
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "google/gemma-4-12b-qat"

SYSTEM_PROMPT = "Ты — полезный ассистент. Отвечай ТОЛЬКО на русском языке. Давай краткие, естественные ответы."

THINKING_PHRASES = [
    "🤔 Думаю над ответом...",
    "💭 Обрабатываю запрос...",
    "⚡ Генерирую ответ...",
    "🧠 Анализирую вопрос...",
]


def get_headers():
    headers = {"Content-Type": "application/json"}
    if os.environ.get("LM_STUDIO_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ.get('LM_STUDIO_API_KEY')}"
    return headers


def trim_history(chat_id, max_messages=10):
    while len(conversation_history[chat_id]) > max_messages:
        if len(conversation_history[chat_id]) > 1:
            conversation_history[chat_id].pop(1)


async def safe_edit(thinking_msg, text):
    """Пытается отредактировать с Markdown, при ошибке форматирования — отправляет как обычный текст."""
    try:
        await thinking_msg.edit_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await thinking_msg.edit_text(text)


async def send_to_model(update: Update, context, user_content):
    """
    Общая функция отправки запроса в LM Studio и обработки ответа.
    user_content может быть строкой (текст) или списком (текст + изображение).
    """
    effective_msg = update.effective_message
    chat_id = effective_msg.chat_id
    thinking_msg = None

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        thinking_msg = await effective_msg.reply_text(random.choice(THINKING_PHRASES))

        if chat_id not in conversation_history:
            conversation_history[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

        conversation_history[chat_id].append({"role": "user", "content": user_content})
        trim_history(chat_id)

        data = {
            "model": MODEL_NAME,
            "messages": conversation_history[chat_id],
            "max_tokens": 4000
        }

        response = requests.post(LM_STUDIO_URL, headers=get_headers(), json=data)

        if response.status_code == 503:
            await safe_edit(thinking_msg, "⚠️ Локальная модель сейчас недоступна. Проверьте, запущен ли сервер LM Studio на порте 1234.")
            return

        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")

        ai_response = response.json()["choices"][0]["message"]["content"]

        if not ai_response:
            ai_response = "🤷 Модель не смогла сформировать ответ, попробуйте переформулировать вопрос."

        conversation_history[chat_id].append({"role": "assistant", "content": ai_response})
        trim_history(chat_id)

        await safe_edit(thinking_msg, ai_response)

    except requests.exceptions.ConnectionError:
        if thinking_msg:
            await safe_edit(thinking_msg, "⚠️ Локальная модель сейчас недоступна. Проверьте, запущен ли сервер LM Studio на порте 1234.")
        else:
            await effective_msg.reply_text("⚠️ Локальная модель сейчас недоступна.")

    except Exception as e:
        if thinking_msg:
            await safe_edit(thinking_msg, f"❌ Ошибка: {str(e)}")
        else:
            await effective_msg.reply_text(f"❌ Ошибка: {str(e)}")


async def handle_message(update: Update, context):
    """Обрабатывает текстовые сообщения пользователя"""
    effective_msg = update.effective_message
    if not effective_msg:
        return

    user_input = effective_msg.text
    await send_to_model(update, context, user_input)


async def handle_photo(update: Update, context):
    """Обрабатывает изображения, отправленные пользователем"""
    effective_msg = update.effective_message
    if not effective_msg or not effective_msg.photo:
        return

    try:
        photo_file = await effective_msg.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        image_b64 = base64.b64encode(photo_bytes).decode("utf-8")

        caption = effective_msg.caption if effective_msg.caption else "Опиши, что на этой картинке"

        user_content = [
            {"type": "text", "text": caption},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]

        await send_to_model(update, context, user_content)

    except Exception as e:
        await effective_msg.reply_text(f"❌ Ошибка при обработке изображения: {str(e)}")


async def start(update: Update, context):
    """Обрабатывает команду /start"""
    try:
        if not update.effective_message:
            return
        await update.effective_message.reply_text(
            "👋 Привет! Я локальный AI-ассистент.\n\n"
            "💬 Просто напиши мне что-нибудь, и я отвечу.\n"
            "🖼 Можешь отправить картинку — я её опишу.\n\n"
            "Команды:\n"
            "🔄 /reset — очистить историю диалога\n"
            "ℹ️ /help — список возможностей"
        )
    except Exception as e:
        print(f"Ошибка в start: {e}")


async def help_command(update: Update, context):
    """Обрабатывает команду /help"""
    try:
        if not update.effective_message:
            return
        await update.effective_message.reply_text(
            "ℹ️ *Что я умею:*\n\n"
            "💬 Отвечать на вопросы и поддерживать диалог\n"
            "🖼 Анализировать изображения (отправь фото, можно с подписью)\n"
            "🧠 Помнить контекст текущей переписки\n\n"
            "*Команды:*\n"
            "🔄 /reset — очистить историю диалога\n"
            "ℹ️ /help — показать это сообщение",
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        print(f"Ошибка в help_command: {e}")


async def reset_history(update: Update, context):
    """Очищает историю диалога"""
    try:
        effective_msg = update.effective_message
        if not effective_msg:
            return

        chat_id = effective_msg.chat_id
        conversation_history[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        await effective_msg.reply_text("🔄 История диалога очищена! Начинаем с чистого листа.")
    except Exception as e:
        print(f"Ошибка в reset_history: {e}")


if __name__ == "__main__":
    try:
        if not TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN не установлен")

        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("reset", reset_history))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        print("Бот запущен. Ожидание сообщений...")
        application.run_polling()

    except Exception as e:
        print(f"Критическая ошибка: {e}")