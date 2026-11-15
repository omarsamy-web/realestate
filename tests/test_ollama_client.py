"""Checks that the Ollama mode code path works when configured."""
from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from unittest.mock import patch

from src import config
from src.indexer import build_index
from src.loaders import load_all
from src.memory import BufferMemory
from src.rag import RealEstateAssistant


class FakeOllamaChat(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake_ollama"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatResult:
        system = messages[0].content if messages else ""
        content = "standalone-query" if "standalone search query" in system else "The down payment is 10%."
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def test_ollama_mode_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(config, "INDEX_DIR", str(tmp_path))

    fake_llm = FakeOllamaChat()
    with patch("src.llm.get_llm", return_value=fake_llm):
        store = build_index(load_all(), offline=True)
        bot = RealEstateAssistant(store, fake_llm, BufferMemory())
        a1, docs, _ = bot.ask("What is the down payment for Palm Residence?")
        a2, _, standalone = bot.ask("And the maintenance deposit?")

        assert a1 == "The down payment is 10%."
        assert a2 == "The down payment is 10%."
        assert standalone == "standalone-query"
