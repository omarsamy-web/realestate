"""Offline stand-ins so the whole pipeline can be tested with NO API key and NO downloads.
Retrieval is lexical (hashed bag-of-words) and the 'LLM' just echoes the top context.
Use it to check wiring and retrieval; use the real models for answer quality."""
import hashlib
import re

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

DIM = 1024


class HashEmbeddings(Embeddings):
    def _vec(self, text: str):
        v = np.zeros(DIM, dtype="float32")
        for tok in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _fake_llm(prompt_value):
    msgs = prompt_value.to_messages() if hasattr(prompt_value, "to_messages") else None
    if msgs is None:
        return AIMessage("(offline summary)")
    system, question = msgs[0].content, msgs[-1].content
    if "standalone search query" in system:
        return AIMessage(question)
    live = ""
    if "Live inventory (from the units database):\n" in system:
        live = system.split("Live inventory (from the units database):\n", 1)[1].split("\n\nContext:", 1)[0]
        live = "" if live.startswith("(no matching units)") else live + "\n"
    ctx = system.split("Context:\n", 1)[-1]
    return AIMessage("[OFFLINE MODE - top retrieved context]\n" + live + ctx[:900])


def OfflineLLM():
    return RunnableLambda(_fake_llm)
