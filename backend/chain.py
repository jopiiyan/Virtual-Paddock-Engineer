"""Phase 2 — RAG chain (standalone, terminal-runnable).

retrieve (k=4) → strict-grounding prompt → ChatOllama → string answer.

    python -m backend.chain "which driver had the worst tyre degradation?"
    python -m backend.chain          # then type a question at the prompt

Note: StrOutputParser() discards the retrieved source docs. Phase 3 (the API)
needs them for "receipts," so it will retrieve separately rather than reuse this
exact chain.
"""

import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from backend.vectorstore import get_vector_store

RETRIEVAL_K = 4

PROMPT = ChatPromptTemplate.from_template(
    "You are a concise Formula 1 race engineer. Answer the question using ONLY the "
    "stint data in the context below. If the context does not contain the answer, say "
    "so plainly (e.g. \"That isn't in the data I have.\") and do not guess or use outside "
    "knowledge.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def build_chain():
    retriever = get_vector_store().as_retriever(search_kwargs={"k": RETRIEVAL_K})
    llm = ChatOllama(model="llama3.2", temperature=0)
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
