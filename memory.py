import logging
from datetime import datetime
import chromadb
from config import get_llm, CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

# Initialize ChromaDB client
_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
_collection = _client.get_or_create_collection(
    name="long_term_memory",
    metadata={"hnsw:space": "cosine"}
)


def extract_key_facts(report: str, query: str) -> list[str]:
    """Use the LLM to extract 3-5 key facts from a research report."""
    llm = get_llm()
    response = llm.invoke(
        f"Extract 3-5 key facts from this research report as short, standalone sentences.\n"
        f"Each fact should make sense on its own without additional context.\n"
        f"Return ONLY the facts, one per line, no numbering, no bullets.\n\n"
        f"Original question: {query}\n\n"
        f"Report:\n{report[:3000]}"
    )
    facts = [
        line.strip().lstrip("*-•0123456789. ")
        for line in response.content.strip().split("\n")
        if line.strip() and len(line.strip()) > 10
    ]
    return facts[:5]


def store_facts(facts: list[str], thread_id: str, query: str):
    """Store extracted facts as embeddings in ChromaDB."""
    if not facts:
        return

    ids = [f"{thread_id}_{i}" for i in range(len(facts))]
    metadatas = [
        {
            "thread_id": thread_id,
            "timestamp": datetime.now().isoformat(),
            "original_query": query
        }
        for _ in facts
    ]

    _collection.upsert(
        ids=ids,
        documents=facts,
        metadatas=metadatas
    )
    logger.info(f"Stored {len(facts)} facts for thread {thread_id[:8]}")


def store_research_memory(report: str, query: str, thread_id: str):
    """Extract key facts from a report and store them in long-term memory."""
    try:
        facts = extract_key_facts(report, query)
        store_facts(facts, thread_id, query)
    except Exception as e:
        logger.warning(f"Failed to store research memory: {e}")


def recall_relevant_facts(query: str, n_results: int = 5, min_similarity: float = 0.7) -> list[str]:
    """Query ChromaDB for facts relevant to the given query.

    Returns a list of fact strings that exceed the similarity threshold.
    """
    try:
        if _collection.count() == 0:
            return []

        results = _collection.query(
            query_texts=[query],
            n_results=min(n_results, _collection.count())
        )

        facts = []
        if results and results["documents"] and results["distances"]:
            for doc, distance in zip(results["documents"][0], results["distances"][0]):
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity: 1 - (distance / 2)
                similarity = 1 - (distance / 2)
                if similarity >= min_similarity:
                    facts.append(doc)

        if facts:
            logger.info(f"Recalled {len(facts)} relevant facts for query: {query[:50]}")
        return facts

    except Exception as e:
        logger.warning(f"Failed to recall facts: {e}")
        return []
