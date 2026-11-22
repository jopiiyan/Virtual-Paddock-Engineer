"""RAG chain (standalone, terminal-runnable).
"""

import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from backend.models import get_llm

from backend.vectorstore import get_vector_store

# Chat retrieves 12 docs (was 4): whole-field questions (podium, team points, rankings)
# need many result docs or the race-summary doc, and 4 regularly missed them.
# Result docs are short, so the extra context is cheap. (The eval harness uses its
# own k from backend/retrieval/config.py — this doesn't change eval numbers.)
RETRIEVAL_K = 12

# Shared grounding instruction — reused by the stateless script (PROMPT below),
# the history-aware prompt in the API, and eval/run_eval.py's answer generation.
GROUNDING_INSTRUCTION = (
    "You are a concise Formula 1 race engineer. Answer using ONLY the race data in the "
    "context below: race summaries (podium, classification, points by team, fastest lap), "
    "per-driver race results, and per-stint data (lap times, sectors, speed traps, tyre "
    "compounds and degradation). You MAY compare, rank, add up and reason over those numbers "
    "to explain things like who was faster and why (e.g. stronger in a sector, higher trap "
    "speed, lower degradation). Do not bring in any knowledge beyond the context. If the "
    "context answers only part of the question, answer that part and say briefly what's "
    "missing; only if it contains nothing relevant, say so plainly (e.g. \"That isn't in the "
    "data I have.\"). Ground every claim in specific facts or numbers from the context."
)

PROMPT = ChatPromptTemplate.from_template(
    GROUNDING_INSTRUCTION + "\n\nContext:\n{context}\n\nQuestion: {question}"
)


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs) #become one block of text


def build_chain():
    retriever = get_vector_store().as_retriever(search_kwargs={"k": RETRIEVAL_K})
    llm = get_llm() #temperature = 0, be as deterministic as possible
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or input("Question: ").strip()
    if not question:
        print("No question provided.")
        return
    print(build_chain().invoke(question))


if __name__ == "__main__":
    main()
