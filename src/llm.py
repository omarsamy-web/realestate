import json
import urllib.error
import urllib.request
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from . import config


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, ChatMessage):
        role = message.role
    else:
        role = "user"
    return {"role": role, "content": str(message.content)}


class ChatOpenAICompatible(BaseChatModel):
    """LangChain chat model wrapper compatible with OpenAI, Groq, BigModel,
    DeepSeek, OpenRouter, Gemini, and any OpenAI-compatible API.

    Works using the official `openai` package if installed, and automatically
    falls back to Python's built-in `urllib.request` if `openai` is not installed."""

    model_name: str = Field(default_factory=lambda: config.LLM_MODEL)
    api_key: str = Field(default_factory=lambda: config.LLM_API_KEY)
    base_url: Optional[str] = Field(default_factory=lambda: config.LLM_BASE_URL)
    temperature: float = 0.0
    timeout: float = 60.0

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

    def _call_via_openai(self, payload: dict) -> str:
        import openai
        client = openai.OpenAI(
            api_key=self.api_key or "not-needed",
            base_url=self.base_url,
            timeout=self.timeout,
        )
        resp = client.chat.completions.create(**payload)
        return resp.choices[0].message.content or ""

    def _call_via_urllib(self, payload: dict) -> str:
        endpoint = (self.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RealEstateRAG/1.0",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, headers=headers, data=data)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"].get("content", "") or ""
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API Error {e.code} from {endpoint}:\n{error_body}") from e

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        formatted_messages = [_convert_message_to_dict(m) for m in messages]
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature,
            **kwargs,
        }
        if stop:
            payload["stop"] = stop

        try:
            content = self._call_via_openai(payload)
        except ImportError:
            content = self._call_via_urllib(payload)

        gen = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[gen])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        formatted_messages = [_convert_message_to_dict(m) for m in messages]
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "stream": True,
            **kwargs,
        }
        if stop:
            payload["stop"] = stop

        try:
            import openai
            client = openai.OpenAI(
                api_key=self.api_key or "not-needed",
                base_url=self.base_url,
                timeout=self.timeout,
            )
            stream = client.chat.completions.create(**payload)
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = delta.content or ""
                if content:
                    gen_chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
                    if run_manager:
                        run_manager.on_llm_new_token(content)
                    yield gen_chunk
        except ImportError:
            endpoint = (self.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RealEstateRAG/1.0",
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(endpoint, headers=headers, data=data)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.strip()
                    if not line or not line.startswith(b"data: "):
                        continue
                    piece = line[6:].strip()
                    if piece == b"[DONE]":
                        break
                    try:
                        msg = json.loads(piece)
                        choices = msg.get("choices", [])
                        if choices:
                            content = choices[0].get("delta", {}).get("content", "")
                            if content:
                                gen_chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
                                if run_manager:
                                    run_manager.on_llm_new_token(content)
                                yield gen_chunk
                    except json.JSONDecodeError:
                        continue


def get_llm(offline: bool = False):
    if offline:
        from .offline import OfflineLLM
        return OfflineLLM()

    if config.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        kwargs = {}
        if config.OLLAMA_API_KEY:
            kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {config.OLLAMA_API_KEY}"}}
        if config.LLM_REASONING == "off":
            kwargs["reasoning"] = False
        return ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_HOST,
            temperature=0,
            num_ctx=config.LLM_NUM_CTX,
            **kwargs,
        )

    # Cloud API providers (Groq, OpenAI, BigModel, DeepSeek, OpenRouter, Gemini, or custom)
    return ChatOpenAICompatible(
        model_name=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=0.0,
    )


def check_llm() -> None:
    """Fail early with a clear message if configuration is missing."""
    if config.LLM_PROVIDER != "ollama":
        if not config.LLM_API_KEY:
            raise SystemExit(
                f"Missing API key for provider '{config.LLM_PROVIDER}'.\n"
                "Please add your API key to .env, e.g.:\n"
                "  LLM_API_KEY=your_key_here\n"
                f"or specify a provider (groq, openai, bigmodel, openrouter, deepseek, gemini)."
            )
        return

    # Ollama local check
    if config.OLLAMA_API_KEY:
        return  # cloud mode: nothing to check locally
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_HOST}/api/tags", timeout=3) as r:
            installed = [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        raise SystemExit(
            f"Cannot reach Ollama at {config.OLLAMA_HOST}.\n"
            "If you want to run with an API instead of downloading Ollama, add to .env:\n"
            "  LLM_PROVIDER=groq  (or openai / bigmodel / deepseek)\n"
            "  LLM_API_KEY=your_key"
        )
    wanted = config.LLM_MODEL if ":" in config.LLM_MODEL else config.LLM_MODEL + ":latest"
    if wanted not in installed:
        raise SystemExit(f"Model '{config.LLM_MODEL}' is not downloaded.\nRun:  ollama pull {config.LLM_MODEL}")


# Backward-compatibility alias
check_ollama = check_llm
