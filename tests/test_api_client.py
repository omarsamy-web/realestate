"""Checks that the API client (ChatOpenAICompatible) builds correct requests
and parses replies, using mocks (no real API calls or downloads needed)."""
from unittest.mock import MagicMock, patch

from src import config
from src.indexer import build_index
from src.llm import ChatOpenAICompatible, get_llm
from src.loaders import load_all
from src.memory import BufferMemory
from src.rag import RealEstateAssistant


def test_api_client_formatting(monkeypatch):
    """Verify that messages are converted to OpenAI format and responses are parsed."""
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key-123")

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "Test API response"

    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_resp) as mock_create:
        llm = get_llm(offline=False)
        out = llm.invoke("Hello, assistant!")
        assert out.content == "Test API response"
        assert mock_create.called
        kwargs = mock_create.call_args.kwargs
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["messages"] == [{"role": "user", "content": "Hello, assistant!"}]


def test_api_client_end_to_end(tmp_path, monkeypatch):
    """Test full RAG assistant pipeline with rewritten query and answer generation via API client."""
    monkeypatch.setattr(config, "INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(config, "LLM_PROVIDER", "bigmodel")
    monkeypatch.setattr(config, "LLM_MODEL", "glm-4-flash")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-api-key")

    calls = []

    def fake_create(*args, **kwargs):
        calls.append(kwargs)
        messages = kwargs.get("messages", [])
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        content = "standalone-query" if "standalone search query" in system else "The down payment is 10%."
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        return mock_resp

    with patch("openai.resources.chat.completions.Completions.create", side_effect=fake_create):
        store = build_index(load_all(), offline=True)
        bot = RealEstateAssistant(store, get_llm(offline=False), BufferMemory())
        a1, docs, _ = bot.ask("What is the down payment for Palm Residence?")
        a2, _, standalone = bot.ask("And the maintenance deposit?")

        assert a1 == "The down payment is 10%."
        assert standalone == "standalone-query"
        assert len(calls) >= 2
        assert any(c["model"] == "glm-4-flash" for c in calls)

