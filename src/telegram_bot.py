"""Telegram price-group listener (Phase 2).

Add the bot to the group where daily prices are posted. Every text message,
caption, or .xlsx/.csv file in an allowed group is parsed and applied to the
units table, then the bot replies with a short summary.

Setup:
  1. Create a bot with @BotFather, put the token in TELEGRAM_BOT_TOKEN.
  2. In @BotFather: /setprivacy -> Disable, so the bot can read all group messages
     (or make the bot a group admin).
  3. Add the bot to the group and send /chatid. Put the number it replies with
     in TELEGRAM_PRICE_CHAT_IDS (comma-separated for several groups).
  4. Run:  python main.py telegram
Uses long polling, so it needs no public URL and runs on any machine.
"""
from __future__ import annotations

import logging
import time

import httpx

from . import config, price_ingest, radar

logger = logging.getLogger(__name__)
API = "https://api.telegram.org"
SHEET_EXTENSIONS = (".xlsx", ".xls", ".csv")


class TelegramClient:
    def __init__(self, token: str, http: httpx.Client | None = None):
        if not token:
            raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env (create a bot with @BotFather).")
        self.token = token
        self.http = http or httpx.Client(timeout=60)

    def call(self, method: str, **params):
        r = self.http.post(f"{API}/bot{self.token}/{method}", json=params)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram {method} failed: {data}")
        return data["result"]

    def download(self, file_id: str) -> bytes:
        info = self.call("getFile", file_id=file_id)
        r = self.http.get(f"{API}/file/bot{self.token}/{info['file_path']}")
        r.raise_for_status()
        return r.content

    def reply(self, chat_id, message_id, text: str):
        try:
            self.call("sendMessage", chat_id=chat_id, text=text[:4000], reply_to_message_id=message_id,
                      allow_sending_without_reply=True)
        except Exception:
            logger.exception("Could not reply in chat %s", chat_id)


def _llm():
    if not config.INGEST_USE_LLM_FALLBACK or config.OFFLINE_MODE:
        return None
    try:
        from .llm import get_llm
        if config.LLM_PROVIDER != "ollama" and not config.LLM_API_KEY:
            return None
        return get_llm()
    except Exception:
        logger.warning("LLM fallback unavailable; using the rule-based parser only")
        return None


def handle_message(client: TelegramClient, msg: dict, llm=None) -> price_ingest.IngestResult | None:
    """Process one Telegram message. Returns the ingest result, or None if ignored."""
    chat_id = str(msg.get("chat", {}).get("id"))
    message_id = msg.get("message_id")
    text = msg.get("text") or msg.get("caption") or ""

    if text.strip().startswith("/chatid"):
        client.reply(chat_id, message_id, f"chat id: {chat_id}")
        return None
    if chat_id not in config.TELEGRAM_PRICE_CHAT_IDS:
        return None

    key = f"telegram:{chat_id}:{message_id}"
    source = f"telegram:{msg.get('chat', {}).get('title') or chat_id}"
    doc = msg.get("document")
    try:
        if doc and str(doc.get("file_name", "")).lower().endswith(SHEET_EXTENSIONS):
            data = client.download(doc["file_id"])
            result = price_ingest.ingest_file(data, doc["file_name"], source=source, message_key=key)
        elif text.strip():
            result = price_ingest.ingest_text(text, source=source, message_key=key, llm=llm)
        else:
            return None
    except Exception as exc:
        logger.exception("Ingest failed for %s", key)
        client.reply(chat_id, message_id, f"⚠️ ما قدرتش أقرا الرسالة: {exc}")
        return None

    changed = result.applied or result.added or result.held
    if config.TELEGRAM_REPLY_IN_GROUP and (changed or result.rejected) and not result.duplicate:
        client.reply(chat_id, message_id, result.summary("ar"))
    return result


def run_polling(client: TelegramClient | None = None, once: bool = False) -> None:
    radar.init_db()
    client = client or TelegramClient(config.TELEGRAM_BOT_TOKEN)
    if not config.TELEGRAM_PRICE_CHAT_IDS:
        logger.warning("TELEGRAM_PRICE_CHAT_IDS is empty: send /chatid in your price group, then set it in .env")
    llm = _llm()
    offset = None
    logger.info("Telegram price listener started")
    while True:
        try:
            params = {"timeout": 50, "allowed_updates": ["message", "channel_post"]}
            if offset is not None:
                params["offset"] = offset
            for update in client.call("getUpdates", **params):
                offset = update["update_id"] + 1
                msg = update.get("message") or update.get("channel_post")
                if msg:
                    handle_message(client, msg, llm)
        except KeyboardInterrupt:
            break
        except Exception:
            logger.exception("Polling error; retrying in 5s")
            time.sleep(5)
        if once:
            break
