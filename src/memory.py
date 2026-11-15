"""Step 4 - Conversation memory.

BufferMemory         : keeps the full chat history.
SummaryBufferMemory  : keeps the latest turns verbatim and folds older turns
                       into a running summary (for long conversations).
"""
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class BufferMemory:
    def __init__(self):
        self.messages: list[BaseMessage] = []

    def add_turn(self, question: str, answer: str) -> None:
        self.messages += [HumanMessage(question), AIMessage(answer)]

    def history(self) -> list[BaseMessage]:
        return list(self.messages)

    def summary(self) -> str:
        return ""

    def clear(self) -> None:
        self.messages = []


class SummaryBufferMemory(BufferMemory):
    def __init__(self, llm, max_turns: int = 6, keep_turns: int = 3):
        super().__init__()
        self.llm = llm
        self.max_turns = max_turns
        self.keep_turns = keep_turns
        self._summary = ""

    def add_turn(self, question: str, answer: str) -> None:
        super().add_turn(question, answer)
        if len(self.messages) // 2 > self.max_turns:
            self._compress()

    def _compress(self) -> None:
        cut = len(self.messages) - self.keep_turns * 2
        old, self.messages = self.messages[:cut], self.messages[cut:]
        transcript = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in old
        )
        prompt = (
            "Update the running summary of a real-estate chat. Keep user preferences "
            "(budget, area, bedrooms, payment plan), units discussed and open questions. "
            "Be concise.\n\n"
            f"Current summary:\n{self._summary or '(none)'}\n\nNew messages:\n{transcript}\n\nUpdated summary:"
        )
        reply = self.llm.invoke(prompt)
        self._summary = getattr(reply, "content", str(reply)).strip()

    def summary(self) -> str:
        return self._summary

    def clear(self) -> None:
        super().clear()
        self._summary = ""
