"""Telegram price-group listener (Phase 2).

Add the bot to the group where daily prices and files are posted.
* Text messages: unit price lines are applied to the units table.
* Any file (PDF, Excel, CSV, Word, TXT, MD, JSON): saved to data/documents/<date>/,
  added to the RAG knowledge base, and any unit prices inside it are applied.
The bot replies with a short summary. Commands: /help, /status, /chatid.

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


HELP_TEXT = """أنا بوت الأسعار والمستندات 🏠
• ابعت رسالة أسعار زي:  PR-101 3,350,000  أو  NV-102 محجوزة  أو  زيادة 5% على Palm Residence
• ابعت أي ملف: PDF، Excel، CSV، Word، TXT. هحفظه وأضيفه لقاعدة المعرفة، ولو فيه أسعار وحدات هحدّثها.
/status  آخر التحديثات
/chatid  رقم الجروب (للإعداد)"""


def status_text() -> str:
    from . import documents
    h = radar.rows("""SELECT (SELECT COUNT(*) FROM units WHERE status='Available') available,
                             (SELECT COUNT(*) FROM units) total,
                             (SELECT MAX(changed_at) FROM price_history) last_change,
                             (SELECT COUNT(*) FROM pending_price_updates WHERE status='Pending') pending""")[0]
    docs = documents.list_documents(5)
    lines = [f"🏠 الوحدات المتاحة: {h['available']} من {h['total']}",
             f"🕒 آخر تغيير سعر: {h['last_change'] or '-'}",
             f"⏸️ تغييرات مستنية موافقة: {h['pending']}"]
    if docs:
        lines.append("📄 آخر الملفات:")
        lines += [f"  • {d['filename']} ({d['created_at'][:16].replace('T', ' ')})" for d in docs]
    return "\n".join(lines)


def handle_message(client: TelegramClient, msg: dict, llm=None):
    """Process one Telegram message. Returns the ingest result, or None if ignored."""
    from . import documents

    chat_id = str(msg.get("chat", {}).get("id"))
    message_id = msg.get("message_id")
    text = msg.get("text") or msg.get("caption") or ""
    command = text.strip().split()[0].split("@")[0].lower() if text.strip().startswith("/") else ""

    if command == "/chatid":
        client.reply(chat_id, message_id, f"chat id: {chat_id}")
        return None
    if chat_id not in config.TELEGRAM_PRICE_CHAT_IDS:
        return None
    if command in ("/help", "/start"):
        client.reply(chat_id, message_id, HELP_TEXT)
        return None
    if command == "/status":
        client.reply(chat_id, message_id, status_text())
        return None

    key = f"telegram:{chat_id}:{message_id}"
    source = f"telegram:{msg.get('chat', {}).get('title') or chat_id}"
    doc = msg.get("document")
    try:
        if doc:
            name = doc.get("file_name") or "document"
            try:
                documents.kind_of(name)
            except documents.UnsupportedDocument:
                client.reply(chat_id, message_id, f"⚠️ مش بقرا النوع ده. ابعت: {', '.join(documents.SUPPORTED)}")
                return None
            if doc.get("file_size", 0) > config.MAX_DOCUMENT_BYTES:
                client.reply(chat_id, message_id, "⚠️ الملف أكبر من 20 ميجا، وتليجرام مش بيسمح للبوت ينزّله.")
                return None
            result = documents.ingest_document(client.download(doc["file_id"]), name, source=source, message_key=key)
            if not result.duplicate:
                client.reply(chat_id, message_id, result.summary("ar"))
            return result
        if msg.get("photo"):
            client.reply(chat_id, message_id, "📷 الصور مش بتتقري لسه. ابعتها كملف PDF أو Excel.")
            return None
        if text.strip() and not command:
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
