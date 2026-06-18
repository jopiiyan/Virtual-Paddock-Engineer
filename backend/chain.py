"""RAG chain (standalone, terminal-runnable).
"""

import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from backend.vectorstore import get_vector_store

RETRIEVAL_K = 4

# Shared grounding instruction — reused by the stateless script (PROMPT below)
# and the history-aware prompt in the API.
GROUNDING_INSTRUCTION = (
    "You are a concise Formula 1 race engineer. Answer using ONLY the stint data in the "
    "context below. You MAY compare, rank, and reason over those numbers — lap times, "
    "sector times, speed-trap figures, tyre degradation and compounds — to explain things "
    "like who was faster and why (e.g. stronger in a sector, higher trap speed, lower "
    "degradation). Do not bring in any knowledge beyond the context; if the context lacks "
    "what's needed to answer, say so plainly (e.g. \"That isn't in the data I have.\"). "
    "Ground every claim in specific numbers from the context."
)

PROMPT = ChatPromptTemplate.from_template(
    GROUNDING_INSTRUCTION + "\n\nContext:\n{context}\n\nQuestion: {question}"
)


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs) #become one block of text


def build_chain():
    retriever = get_vector_store().as_retriever(search_kwargs={"k": RETRIEVAL_K})
    llm = ChatOllama(model="llama3.2", temperature=0) #temperature = 0, be as deterministic as possible
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
