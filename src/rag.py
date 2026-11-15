"""Step 3 - RAG pipeline with memory-aware follow-up handling."""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from . import config
from .memory import BufferMemory

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's last message as a standalone search query, using the chat history "
     "to resolve words like 'it', 'that unit', 'the cheaper one'. Keep the user's language. "
     "If it is already standalone, return it unchanged. Output only the query."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a real-estate sales assistant. Answer ONLY from the context below.\n"
     "Rules:\n"
     "- If the answer is not in the context, say you don't have that information. Never guess.\n"
     "- Copy prices, areas, dates and percentages exactly as written.\n"
     "- Cite sources in brackets, e.g. [Palm_Residence_Brochure.pdf p.2] or [units.csv row 5].\n"
     "- Reply in the same language the user writes in (Arabic or English).\n"
     "- Use the conversation summary and history for follow-up questions.\n\n"
     "Conversation summary: {summary}\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def format_context(docs) -> str:
    lines = []
    for d in docs:
        m = d.metadata
        tag = f"{m['source']} p.{m['page']}" if m.get("type") == "pdf" else f"{m['source']} row {m.get('row')}"
        lines.append(f"[{tag}] {d.page_content}")
    return "\n\n".join(lines)


class RealEstateAssistant:
    def __init__(self, vectorstore, llm, memory=None, k: int = config.TOP_K):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        self.memory = memory or BufferMemory()
        self.rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()
        self.answer_chain = ANSWER_PROMPT | llm | StrOutputParser()

    def ask(self, question: str):
        history = self.memory.history()
        # 1) make follow-ups self-contained so retrieval works ("and its payment plan?")
        standalone = (
            self.rewrite_chain.invoke({"history": history, "question": question}).strip()
            if history else question
        )
        # 2) retrieve
        docs = self.retriever.invoke(standalone)
        # 3) answer with context + memory
        answer = self.answer_chain.invoke({
            "summary": self.memory.summary() or "(none)",
            "context": format_context(docs),
            "history": history,
            "question": question,
        })
        # 4) remember
        self.memory.add_turn(question, answer)
        return answer, docs, standalone
