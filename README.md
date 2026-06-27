# Telegram Bot with Local LLM via LM Studio

A Telegram bot that connects to a locally running LLM (via LM Studio) to provide AI-powered chat responses. Supports conversation history, image analysis (with vision-capable models), and a polished UX with typing indicators and emoji feedback.

## Features

- 💬 Text conversations with context memory per chat
- 🖼 Image analysis support (requires a vision-capable model)
- ⌨️ "Typing..." indicator while generating responses
- 🔄 `/reset` command to clear conversation history
- ℹ️ `/help` command listing bot capabilities
- ⚠️ Graceful error handling (e.g. LM Studio server offline)

## Requirements

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) running locally with a model loaded and the local server enabled
- A Telegram bot token from [@BotFather](https://t.me/BotFather)

## Setup

1. Clone the repository:
```bash
   git clone https://github.com/ArtjomsS53/Telegram-bot-with-local-LLM-via-LM-Studio.git
   cd Telegram-bot-with-local-LLM-via-LM-Studio
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
```

4. Make sure LM Studio is running locally with the server enabled (default: `http://localhost:1234`).

5. Update the `MODEL_NAME` variable in the bot script to match the model loaded in LM Studio.

6. Run the bot:
```bash
   python telegram-local-llm-bot.py
```

## Usage

- Send any text message to chat with the model
- Send an image (with an optional caption) to have it analyzed by a vision-capable model
- `/start` — welcome message
- `/reset` — clear conversation history
- `/help` — list available commands

## Notes

- Conversation history is stored in memory and resets when the bot restarts
- Make sure your loaded model fits within your available VRAM for best performance

## 👤 Author

**ArtjomsS53**
Educational project exploring local LLM integration, Telegram bot development, and API communication between Python and LM Studio.
