# Telegram Bot with Local LLM via LM Studio

A Telegram bot that connects to locally running LLMs (via LM Studio) to provide private, fully local AI chat. Supports multiple switchable models, conversation memory, image analysis, voice message transcription, and a document-based RAG memory system — all running on your own hardware. Also includes an optional **Telegram Mini App**: a full web interface with the same chat and RAG memory, opening right inside Telegram.

## Features

### Core Chat
- Text conversations with per-chat memory
- Streaming responses with live progress (token count, speed)
- Switch between multiple local models on the fly (`/model`)
- Image analysis with vision-capable models
- Voice message transcription via Whisper, with confirm/edit/remember/short-answer actions

### RAG Document Memory
- Upload PDF, TXT, MD, LOG, CSV, JSON, or DOCX files — the bot extracts text, chunks it, and stores embeddings locally
- Hybrid search (semantic + keyword) to find relevant context for your questions
- Per-document tools via inline buttons: summary, key points, study breakdown, quiz generation, document comparison, download original, delete
- RAG-only mode — answers strictly from your uploaded documents
- Deduplication by file hash
- Export/import your RAG memory as JSON

### Study Mode
- Toggle a teaching-style response mode with explanations and follow-up questions

### Reliability & Admin
- Per-chat task queue — prevents overlapping heavy operations
- Automatic and manual database backups
- Admin-only commands: usage stats, user list, database size, model sync
- `/status` healthcheck for LM Studio, Whisper, and the RAG database
- Safe Telegram HTML rendering with plain-text fallback

### Telegram Mini App (optional)
- Full web UI for the bot (FastAPI backend + single-page frontend), sharing the same RAG database and LM Studio connection as the Telegram bot
- Streaming chat with multiple saved conversations (sessions)
- Model switcher with background loading
- Document manager: upload / download / delete / preview, right from the browser
- Hybrid RAG search (semantic + keyword) from the UI
- RAG-only and Study mode toggles
- Voice input (Whisper transcription)
- Admin stats screen
- Runs entirely on your own machine — a free Cloudflare tunnel script is included to expose it over HTTPS for Telegram, no hosting or account required
- See [`miniapp/README.md`](miniapp/README.md) for the full guide (screens, setup, Telegram integration)

## Requirements

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) running locally with one or more models loaded and the local server enabled
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- `ffmpeg` installed and available in PATH (required for voice message transcription)

## Setup

1. Clone the repository:
 `git clone https://github.com/ArtjomsS53/Telegram-bot-with-local-LLM-via-LM-Studio.git`

   `cd Telegram-bot-with-local-LLM-via-LM-Studio`

2. Install dependencies:
   `pip install -r requirements.txt`

3. Create a `.env` file in the project root:
   ```bash
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
   ADMIN_TELEGRAM_IDS=your_telegram_user_id
   ADMIN_ONLY_MODE=false
   TELEGRAM_RICH_TEXT=true
   PLAIN_TEXT_OUTPUT=false
   ```

5. Make sure LM Studio is running locally with the server enabled (default: http://localhost:1234).

6. Update the `AVAILABLE_MODELS` dictionary in the bot script to match the models you have loaded in LM Studio.

7. Run the bot:
   `python telegram-local-llm-bot.py`

8. (Optional) Run everything at once — bot + Mini App server + Cloudflare tunnel, each in its own window:
   `start_all.bat` (Windows). Individual scripts are also available: `restart_server.bat` restarts just the Mini App server without dropping the tunnel, and `miniapp\run_miniapp.bat` starts only the Mini App. See [`miniapp/README.md`](miniapp/README.md) for details.

## Usage

### Basic
- Send any text message to chat with the model
- Send an image (with an optional caption) to have it analyzed
- Send a voice message — it will be transcribed and you'll get action buttons
- `/start` — welcome message
- `/reset` — clear conversation history
- `/model` — switch between available models
- `/help` — list all commands
- `/status` — healthcheck

### RAG / Document Memory
- Send a PDF, TXT, MD, LOG, CSV, JSON, or DOCX file to add it to memory
- `/remember <text>` — add raw text to memory
- `/rag_docs` — list documents with action buttons
- `/rag_summary <id|name>` — get a document summary
- `/rag_compare <id1> <id2>` — compare two documents
- `/quiz <id|name>` — generate a quiz from a document
- `/rag_search <query>` — search memory directly
- `/rag_only_on` / `/rag_only_off` — toggle RAG-only mode
- `/rag_delete <id|name>` — delete a document
- `/rag_clear` — wipe all memory for this chat
- `/rag_export` / `/rag_import` — back up or restore your memory

### Study Mode
- `/study_mode_on` / `/study_mode_off` — toggle teaching-style answers

### Admin (requires `ADMIN_TELEGRAM_IDS`)
- `/admin_stats` — bot-wide statistics
- `/admin_users` — recent users
- `/admin_db_size` — database size and path
- `/admin_reload_models` — re-sync LM Studio loaded models
- `/backup` / `/backups` — manage database backups

## Telegram Mini App

An optional local web interface that opens right inside Telegram (or in a regular browser for testing), backed by the same `rag_files/rag_memory.sqlite3` database and LM Studio connection as the bot — nothing about the main bot script changes when you use it.

Quick start:
1. Make sure LM Studio is running with its local server enabled.
2. Run `miniapp\run_miniapp.bat` (installs its dependencies and starts the server), or `start_all.bat` to launch the bot, the Mini App, and a Cloudflare tunnel together.
3. Open `http://localhost:8080` in a browser to try it locally, or expose it over HTTPS with the included tunnel script and set that link as your bot's Mini App URL in [@BotFather](https://t.me/BotFather).

Extra dependencies (FastAPI, uvicorn, etc.) live in `miniapp/requirements-miniapp.txt`. Full setup, screens, and Telegram configuration steps are documented in [`miniapp/README.md`](miniapp/README.md).

## Notes

- Conversation history is stored in memory and resets when the bot restarts
- RAG memory and document files persist on disk (`rag_files/`) and are excluded from version control
- Backups are stored in `backups/` and are also excluded from version control
- Make sure your loaded models fit within your available VRAM for best performance
- Voice transcription runs on CPU by default to leave VRAM free for the LLM
- The Mini App's conversation history is kept separately from the Telegram bot's, but they share the same RAG memory, documents, and per-chat settings (`rag_only`, `study_mode`)
- `miniapp/cloudflared.exe` is not included in the repo (~50MB binary) — download it once per [`miniapp/tunnel_cloudflare.bat`](miniapp/tunnel_cloudflare.bat) instructions

## 👤 Author

**ArtjomsS53**
Educational project exploring local LLM integration, Telegram bot development, RAG memory systems, and API communication between Python and LM Studio.

Built with the help of Claude and ChatGPT for code generation, debugging, and architecture decisions.
