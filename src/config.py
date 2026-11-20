import os
import re
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# Ensure Windows loads package DLLs (such as faiss.dll) correctly
_DLL_HANDLES = []
if hasattr(os, "add_dll_directory"):
    try:
        import site
        for sp in site.getsitepackages():
            faiss_dir = Path(sp) / "faiss"
            if faiss_dir.is_dir():
                _DLL_HANDLES.append(os.add_dll_directory(str(faiss_dir)))
    except Exception:
        pass
try:
    import faiss
except Exception:
    pass

# Provider endpoint defaults
PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "bigmodel": "https://open.bigmodel.cn/api/paas/v4",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama": "http://localhost:11434",
}

PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "openai/gpt-oss-120b",
    "bigmodel": "glm-4-flash",
    "zhipu": "glm-4-flash",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-1.5-flash",
    "ollama": "gemma4:e4b",
}

# Resolve API Key from possible env vars
LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("GROQ_API_KEY")
    or os.getenv("ZHIPUAI_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("OLLAMA_API_KEY")
    or ""
).strip().strip('"').strip("'")


def _detect_provider(key: str) -> str:
    if not key:
        return "openai"
    if key.startswith("gsk_"):
        return "groq"
    if key.startswith("sk-or-"):
        return "openrouter"
    if key.startswith("AIza"):
        return "gemini"
    if re.match(r"^[a-f0-9]{32}\.[a-zA-Z0-9_-]+$", key):
        return "bigmodel"
    return "openai"


# LLM Provider: 'openai' (default for API), 'groq', 'bigmodel', 'openrouter', 'deepseek', 'gemini', 'ollama', etc.
_env_provider = os.getenv("LLM_PROVIDER")
if _env_provider:
    LLM_PROVIDER = _env_provider.strip().lower()
elif LLM_API_KEY:
    LLM_PROVIDER = _detect_provider(LLM_API_KEY)
else:
    LLM_PROVIDER = "openai"

# Model
default_model = PROVIDER_DEFAULT_MODELS.get(LLM_PROVIDER, "gpt-4o-mini")
_env_model = os.getenv("LLM_MODEL", "").strip().strip('"').strip("'")
if not _env_model:
    LLM_MODEL = default_model
elif LLM_PROVIDER in ("bigmodel", "zhipu") and ("gpt-oss" in _env_model or "gemma" in _env_model):
    LLM_MODEL = "glm-4-flash"
elif LLM_PROVIDER == "openai" and ("gpt-oss" in _env_model or "gemma" in _env_model):
    LLM_MODEL = "gpt-4o-mini"
elif LLM_PROVIDER == "groq":
    m_clean = _env_model.lower().replace(":", "-")
    if "gpt-oss-120b" in m_clean:
        LLM_MODEL = "openai/gpt-oss-120b"
    elif "gpt-oss-20b" in m_clean:
        LLM_MODEL = "openai/gpt-oss-20b"
    elif "qwen" in m_clean:
        LLM_MODEL = "qwen/qwen3.8-27b"
    elif "allam" in m_clean:
        LLM_MODEL = "allam-2-7b"
    else:
        LLM_MODEL = _env_model
else:
    LLM_MODEL = _env_model

# Base URL
default_base_url = PROVIDER_BASE_URLS.get(LLM_PROVIDER, "https://api.openai.com/v1")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or default_base_url

# For backwards compatibility with Ollama settings
OLLAMA_API_KEY = LLM_API_KEY
OLLAMA_HOST = os.getenv("OLLAMA_HOST") or LLM_BASE_URL
LLM_REASONING = os.getenv("LLM_REASONING", "off")
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))

# ---- Embeddings: open-source, local, supports Arabic + English ----
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def _resolve_path(path_str: str) -> str:
    if not path_str or "://" in path_str:
        return path_str
    p = Path(path_str)
    if not p.is_absolute():
        p = ROOT_DIR / p
    return str(p)


PDF_DIR = _resolve_path(os.getenv("PDF_DIR", "data/pdfs"))
SHEET_SOURCE = _resolve_path(os.getenv("SHEET_SOURCE", "data/units.csv"))
INDEX_DIR = _resolve_path(os.getenv("INDEX_DIR", "index"))

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
MAX_RAG_SHEET_ROWS = max(1, int(os.getenv("MAX_RAG_SHEET_ROWS", "5000")))

# ---- Research Lab settings (Phase 2) ----
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
LLM_CHUNK_CHAR_LIMIT = int(os.getenv("LLM_CHUNK_CHAR_LIMIT", "24000"))
REPORTS_DIR = _resolve_path(os.getenv("REPORTS_DIR", "generated_reports"))


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# ---- Phase 1: API service ----
DB_PATH = _resolve_path(os.getenv("RADAR_DB", "data/radar.db"))
OFFLINE_MODE = _flag("OFFLINE_MODE")          # hashed embeddings + echo LLM, for testing only
API_KEY = os.getenv("API_KEY", "").strip()    # if set, staff endpoints need header X-API-Key
CHAT_SESSION_TTL_MIN = int(os.getenv("CHAT_SESSION_TTL_MIN", "120"))

# ---- Phase 2: daily price ingestion ----
PRICE_CHANGE_HOLD_PCT = float(os.getenv("PRICE_CHANGE_HOLD_PCT", "25"))  # bigger jumps wait for approval
AUTO_REBUILD_INDEX = _flag("AUTO_REBUILD_INDEX", "1")
INGEST_WEBHOOK_TOKEN = os.getenv("INGEST_WEBHOOK_TOKEN", "").strip()
INGEST_USE_LLM_FALLBACK = _flag("INGEST_USE_LLM_FALLBACK", "1")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_PRICE_CHAT_IDS = {c.strip() for c in os.getenv("TELEGRAM_PRICE_CHAT_IDS", "").split(",") if c.strip()}
TELEGRAM_REPLY_IN_GROUP = _flag("TELEGRAM_REPLY_IN_GROUP", "1")
