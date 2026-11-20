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
import re
import time

import httpx

from . import config, price_ingest, radar

logger = logging.getLogger(__name__)
API = "https://api.telegram.org"


# httpx logs every request URL at INFO, and Telegram URLs contain the bot token
logging.getLogger("httpx").setLevel(logging.WARNING)


class BotConflict(RuntimeError):
    """Another program is already reading updates for this bot token."""


CONFLICT_HELP = (
    "Telegram says another program is already using this bot token (409 Conflict).\n"
    "Only ONE program can read a bot's messages at a time. Fix it by one of:\n"
    "  1. Close any other terminal running 'main.py telegram' (or the old project's bot) with the same token.\n"
    "  2. Make sure TELEGRAM_CLIENT_BOT_TOKEN is a DIFFERENT bot from TELEGRAM_BOT_TOKEN.\n"
    "  3. If the token is used on another computer or server, stop it there, or create a new bot with @BotFather.\n"
    "Retrying every 30 seconds..."
)


class TelegramClient:
    def __init__(self, token: str, http: httpx.Client | None = None):
        if not token:
            raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env (create a bot with @BotFather).")
        self.token = token
        self.http = http or httpx.Client(timeout=60)

    def call(self, method: str, **params):
        r = self.http.post(f"{API}/bot{self.token}/{method}", json=params)
        if r.status_code == 409:
            raise BotConflict(r.json().get("description", "Conflict"))
        if r.status_code == 401:
            raise SystemExit("Telegram rejected the token (401). Check TELEGRAM_BOT_TOKEN in .env.")
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


# --------------------------------------------------------------------------- sales assistant (Phase 4)
SALES_HELP = """مساعد السيلز 🤝 (عنده كل الوحدات)
اكتب سؤالك عادي، زي:
• فلل التجمع تحت 15 مليون
• كام وحدة متاحة في Rai Valleys وأسعارها من كام لكام
• أرخص شقة 3 غرف في زايد
• تاريخ سعر PR-101
أوامر:
/units <بحث>  الوحدات المطابقة من غير ذكاء اصطناعي
/unit <كود>  تفاصيل وحدة وتاريخ سعرها
/projects  ملخص كل المشاريع
/status  حالة الأسعار والملفات
/reset  محادثة جديدة
في الجروب: اعمل منشن للبوت أو رد على رسالته عشان يرد عليك."""

CRM_HELP = """
أوامر الـ CRM:
/priority [عدد]  مين تكلمه النهاردة
/clusters  شرائح العملاء
/lead <كود>  ملف العميل
/recommend <كود>  أنسب الوحدات + رسالة واتساب
/newleads  آخر العملاء من الشات بوت"""

_sales_assistant = {}


def _sales_bot():
    from .sales_units import build_sales_assistant
    if "bot" not in _sales_assistant:
        _sales_assistant["bot"] = build_sales_assistant()
    return _sales_assistant["bot"]


def _sales_answer(question: str, session: str = "default") -> str:
    return _sales_bot().answer(question, session=session)["answer"]


def _addressed_to_bot(msg: dict) -> str | None:
    """In a group: the question text if the bot was @mentioned or replied to, else None."""
    text = (msg.get("text") or "").strip()
    reply_to = (msg.get("reply_to_message") or {}).get("from") or {}
    if reply_to.get("is_bot") and reply_to.get("username") == _me.get("username"):
        return text
    name = _me.get("username")
    if name and f"@{name}".lower() in text.lower():
        return re.sub(rf"@{re.escape(name)}", "", text, flags=re.I).strip()
    return None


_me: dict = {}


def _fmt_profile(p: dict) -> str:
    budget = f"{p['budget']:,.0f}" if p.get("budget") else "?"
    lines = [f"👤 {p.get('name')} ({p['lead_key']}) | أولوية {p.get('priority')} | شريحة {p.get('cluster')}",
             f"📍 {p.get('location') or '?'} | 🏠 {p.get('unit_type') or '?'} | "
             f"{p['bedrooms'] if p.get('bedrooms') is not None else '?'} غرف | 💰 {budget} | 💳 {p.get('payment') or '?'}",
             f"📞 {p.get('phone') or '-'} | المسؤول: {p.get('agent') or '-'} | آخر تواصل: {(p.get('last_contact') or '-')[:10]}",
             f"🏘️ وحدات متاحة مناسبة: {p.get('matching_units') or 0}"]
    if p.get("objections"):
        lines.append("⚠️ اعتراضات: " + "، ".join(p["objections"]))
    if p.get("last_note"):
        lines.append(f"📝 {p['last_note'][:200]}")
    if p.get("priority_reasons"):
        lines.append("ليه الأولوية دي: " + "; ".join(p["priority_reasons"]))
    return "\n".join(lines)


def handle_sales(client: TelegramClient, msg: dict) -> bool:
    """Sales commands. Returns True if the message was handled."""
    from . import lead_clustering, sales_agent, sales_units

    chat_id = str(msg.get("chat", {}).get("id"))
    if chat_id not in config.TELEGRAM_SALES_CHAT_IDS:
        return False
    message_id = msg.get("message_id")
    text = (msg.get("text") or "").strip()
    private = msg.get("chat", {}).get("type") == "private"
    parts = text.split(maxsplit=1)
    command = parts[0].split("@")[0].lower() if text.startswith("/") else ""
    arg = parts[1].strip() if len(parts) > 1 else ""
    reply = None
    crm_commands = ("/priority", "/clusters", "/lead", "/recommend", "/newleads")
    try:
        if command in ("/help", "/start"):
            reply = SALES_HELP + (CRM_HELP if config.SALES_CRM_ACCESS else "")
        elif command in crm_commands and not config.SALES_CRM_ACCESS:
            reply = "الأمر ده مقفول. بوت السيلز عنده وصول للوحدات بس."
        elif command == "/units":
            reply = sales_units.list_units(arg) if arg else "اكتب: /units فلل التجمع تحت 15 مليون"
        elif command == "/unit":
            reply = (sales_units.unit_details(arg) or "مش لاقي الوحدة دي.") if arg else "اكتب: /unit PR-101"
        elif command == "/projects":
            reply = sales_units.projects_overview()
        elif command == "/priority":
            n = int(arg) if arg.isdigit() else 10
            leads = lead_clustering.priority_list(min(n, 30))
            if not leads:
                reply = "مافيش بروفايلات لسه. شغّل:  python main.py profile-leads"
            else:
                reply = "📞 كلّم دول الأول:\n" + "\n".join(
                    f"{i}. {p['name']} ({p['lead_key']}) — {p['priority']} | {p.get('location') or '?'} | "
                    f"{(p['budget'] or 0) / 1e6:.1f}M | {p.get('matching_units') or 0} وحدة مناسبة"
                    for i, p in enumerate(leads, 1)) + "\n\nللتفاصيل: /recommend <الكود>"
        elif command == "/clusters":
            clusters = lead_clustering.list_clusters()
            if not clusters:
                reply = "مافيش شرائح لسه. شغّل:  python main.py profile-leads"
            else:
                reply = "\n\n".join(f"#{c['cluster']} ({c['size']} عميل) {c['label']}\n💡 {c['playbook']}" for c in clusters)
        elif command == "/lead":
            p = lead_clustering.get_profile(arg) if arg else None
            reply = _fmt_profile(p) if p else "مش لاقي العميل ده. جرّب الكود أو رقم الموبايل."
        elif command == "/recommend":
            reply = sales_agent.format_recommendation(sales_agent.recommend_for(arg)) if arg else "اكتب: /recommend <كود العميل>"
        elif command == "/newleads":
            leads = radar.rows("SELECT * FROM leads WHERE notes LIKE 'From client chatbot%' ORDER BY id DESC LIMIT 10")
            reply = "\n".join(f"#{l['id']} {l['name']} | {l['phone']} | {l['area'] or '?'} | "
                               f"{(l['budget'] or 0) / 1e6:.1f}M | {l['created_at'][:16].replace('T', ' ')}"
                               for l in leads) or "مافيش عملاء جدد من الشات بوت."
        elif command == "/status":
            reply = status_text()
        elif command in ("/reset", "/new"):
            _sales_bot().reset(chat_id)
            reply = "تمام، بدأنا محادثة جديدة."
        elif command == "/ask" or (private and text and not command and chat_id not in config.TELEGRAM_PRICE_CHAT_IDS) \
                or (not private and not command and _addressed_to_bot(msg)):
            question = arg if command == "/ask" else (text if private else _addressed_to_bot(msg))
            reply = _sales_answer(question, session=chat_id) if question else "اكتب: /ask <سؤالك>"
        else:
            return False
    except ValueError as e:
        reply = f"⚠️ {e}"
    except Exception as e:
        logger.exception("Sales command failed")
        reply = f"⚠️ حصلت مشكلة: {e}"
    for i in range(0, len(reply), 3900):
        client.reply(chat_id, message_id, reply[i:i + 3900])
    return True


def handle_internal(client: TelegramClient, msg: dict, llm=None):
    """The internal bot serves the price group(s) and the sales team."""
    if handle_sales(client, msg):
        return None
    return handle_message(client, msg, llm)


def sales_notifier(client: TelegramClient | None = None):
    """Function that posts a message to every sales chat (used by the client bot for hand-offs)."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_SALES_CHAT_IDS:
        return lambda text: logger.info("Sales alert (no sales chat configured): %s", text)
    client = client or TelegramClient(config.TELEGRAM_BOT_TOKEN)

    def notify(text: str):
        for chat in config.TELEGRAM_SALES_CHAT_IDS:
            client.call("sendMessage", chat_id=chat, text=text[:4000])
    return notify


# --------------------------------------------------------------------------- client bot (Phase 3)
CLIENT_WELCOME = ("أهلاً بحضرتك 👋 أنا المساعد العقاري.\n"
                  "قولّي بتدوّر على إيه: المنطقة، الميزانية، نوع الوحدة وعدد الغرف، وأنا أرشحلك الوحدات المتاحة بأسعار النهاردة.")


def handle_client_message(client: TelegramClient, msg: dict, agent) -> dict | None:
    chat = msg.get("chat", {})
    if chat.get("type") != "private":
        return None  # the client bot only talks one-to-one
    chat_id, message_id = str(chat.get("id")), msg.get("message_id")
    user = msg.get("from", {})
    name = " ".join(x for x in (user.get("first_name"), user.get("last_name")) if x) or None
    text = (msg.get("text") or "").strip()
    if msg.get("contact"):
        text = f"رقمي {msg['contact'].get('phone_number', '')}"
    if text in ("/start", "/help"):
        client.reply(chat_id, message_id, CLIENT_WELCOME)
        return None
    if text == "/reset":
        agent.reset("telegram", chat_id)
        client.reply(chat_id, message_id, "تمام، نبدأ من الأول. بتدوّر على إيه؟")
        return None
    if not text or text.startswith("/"):
        client.reply(chat_id, message_id, "ابعتلي طلبك في رسالة نصية 🙏")
        return None
    try:
        result = agent.handle("telegram", chat_id, text, name=name)
    except Exception:
        logger.exception("Client agent failed")
        client.reply(chat_id, message_id, "معلش حصلت مشكلة، حد من فريقنا هيتواصل معاك قريب.")
        return None
    client.reply(chat_id, message_id, result["reply"])
    return result


# --------------------------------------------------------------------------- polling
def _poll(client: TelegramClient, handle, name: str, once: bool = False) -> None:
    offset = None
    try:
        me = client.call("getMe")
        _me.update(me if isinstance(me, dict) else {})
        # a webhook left over from another tool blocks polling; long polling needs it removed
        client.call("deleteWebhook", drop_pending_updates=False)
        logger.info("%s started as @%s", name, me.get("username"))
    except BotConflict:
        pass
    warned = False
    while True:
        try:
            params = {"timeout": 50, "allowed_updates": ["message", "channel_post"]}
            if offset is not None:
                params["offset"] = offset
            for update in client.call("getUpdates", **params):
                offset = update["update_id"] + 1
                msg = update.get("message") or update.get("channel_post")
                if msg:
                    handle(msg)
        except KeyboardInterrupt:
            break
        except BotConflict:
            if not warned:
                logger.error(CONFLICT_HELP)
                warned = True
            time.sleep(30)
        except Exception as exc:
            logger.error("Polling error (%s); retrying in 5s", type(exc).__name__)
            time.sleep(5)
        if once:
            break


def run_polling(client: TelegramClient | None = None, once: bool = False) -> None:
    """Internal bot: price groups + sales team."""
    radar.init_db()
    client = client or TelegramClient(config.TELEGRAM_BOT_TOKEN)
    if not config.TELEGRAM_PRICE_CHAT_IDS and not config.TELEGRAM_SALES_CHAT_IDS:
        logger.warning("Set TELEGRAM_PRICE_CHAT_IDS / TELEGRAM_SALES_CHAT_IDS: send /chatid to the bot to get the id")
    llm = _llm()
    _poll(client, lambda m: handle_internal(client, m, llm), "Telegram internal bot (prices + sales)", once)


def run_client_bot(client: TelegramClient | None = None, once: bool = False) -> None:
    """Client bot: a separate bot token that buyers talk to privately."""
    from .client_agent import build_client_agent
    if not config.TELEGRAM_CLIENT_BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_CLIENT_BOT_TOKEN in .env (a second bot from @BotFather, not the internal one).")
    if config.TELEGRAM_CLIENT_BOT_TOKEN == config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_CLIENT_BOT_TOKEN must be a different bot from TELEGRAM_BOT_TOKEN.")
    radar.init_db()
    client = client or TelegramClient(config.TELEGRAM_CLIENT_BOT_TOKEN)
    # fully separate by default: no CRM, no messages to the sales bot
    agent = build_client_agent(notify=sales_notifier() if config.CLIENT_SAVE_LEADS else None)
    _poll(client, lambda m: handle_client_message(client, m, agent), "Telegram client bot", once)
