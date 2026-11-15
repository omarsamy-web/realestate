"""Step 5 - Testing and evaluation.

Metrics
  retrieval hit   : was the expected source file among the retrieved chunks?
  context recall  : do the retrieved chunks contain every expected keyword?
  answer accuracy : does the answer contain every expected keyword? (refusal cases: does it admit it doesn't know?)
  conversation    : same accuracy check across multi-turn chats, to test memory and follow-ups

Run:  python -m eval.evaluate            (real models)
      python -m eval.evaluate --offline  (retrieval metrics only are meaningful)
"""
import argparse
import csv
import json
from pathlib import Path

from src.indexer import load_index
from src.llm import check_llm, get_llm
from src.memory import BufferMemory
from src.rag import RealEstateAssistant


def norm(text: str) -> str:
    return text.lower().replace(",", "").replace("\u2019", "'")


def has_all(text: str, keywords) -> bool:
    return all(norm(k) in norm(text) for k in keywords)


def refuses(text: str) -> bool:
    t = norm(text)
    return any(p in t for p in ["don't have", "do not have", "not available in", "no information", "not mentioned", "cannot find", "can't find"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    if not args.offline:
        check_llm()
    tests = json.loads(Path(__file__).with_name("test_set.json").read_text())
    store, llm = load_index(args.offline), get_llm(args.offline)
    rows, hits, recalls, correct = [], [], [], []

    for t in tests["single_turn"]:
        bot = RealEstateAssistant(store, llm, BufferMemory())  # fresh memory per question
        answer, docs, _ = bot.ask(t["q"])
        retrieved = "\n".join(d.page_content for d in docs)
        if t.get("expect_refusal"):
            ok, hit, rec = refuses(answer), None, None
        else:
            hit = t["source"] in {d.metadata["source"] for d in docs}
            rec = has_all(retrieved, t["keywords"])
            ok = has_all(answer, t["keywords"])
            hits.append(hit); recalls.append(rec)
        correct.append(ok)
        rows.append(["single", t["q"], hit, rec, ok])

    conv_ok = []
    for conv in tests["conversations"]:
        bot = RealEstateAssistant(store, llm, BufferMemory())
        for turn in conv["turns"]:
            answer, _, _ = bot.ask(turn["q"])
            ok = has_all(answer, turn["keywords"])
            conv_ok.append(ok)
            rows.append([f"conv: {conv['name']}", turn["q"], None, None, ok])

    pct = lambda xs: f"{100 * sum(xs) / len(xs):.0f}% ({sum(xs)}/{len(xs)})" if xs else "n/a"
    out = Path(__file__).with_name("results.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["type", "question", "retrieval_hit", "context_recall", "answer_ok"]); w.writerows(rows)

    print("\n=== Evaluation ===")
    print(f"Retrieval hit (right source retrieved): {pct(hits)}")
    print(f"Context recall (facts in retrieved text): {pct(recalls)}")
    print(f"Answer accuracy, single questions:        {pct(correct)}")
    print(f"Answer accuracy, multi-turn follow-ups:   {pct(conv_ok)}")
    if args.offline:
        print("\nOFFLINE MODE: answer metrics are not meaningful (the fake LLM only echoes context).")
    print(f"Details saved to {out}")


if __name__ == "__main__":
    main()
