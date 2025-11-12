import os
import json
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from schemas import RAGAnswer, Source
from dotenv import load_dotenv
from duckduckgo_search import DDGS

# Load environment variables
load_dotenv()


# -----------------------------
# Helper: Load all vectorstores
# -----------------------------
def load_all_vectorstores() -> List[Chroma]:
    """Loads all available Chroma domain collections."""
    persist_directory = os.path.join(os.getcwd(), "data", "chroma")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Define domain-specific collections
    domain_collections = ["finance", "hr", "sustainability", "general"]
    stores = []

    for domain in domain_collections:
        collection_name = f"docs_{domain}"
        try:
            store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory,
            )
            stores.append((domain, store))
            print(f"[INFO] Loaded collection '{collection_name}'")
        except Exception as e:
            print(f"[WARN] Could not load collection '{collection_name}': {e}")
    return stores


# -----------------------------
# Helper: System Prompt
# -----------------------------
SYSTEM_PROMPT = """
You are a careful analyst that answers questions using ONLY the provided context.
If you cannot find a complete answer, list what is missing in the 'missing_info' field.

Respond strictly in the following JSON format:
{
  "answer": "string",
  "confidence": float,
  "missing_info": ["string"],
  "citations": [
      {"doc_id": "string", "filename": "string", "page": int, "score": float, "snippet": "string"}
  ],
  "reasoning_summary": "string",
  "suggestions": ["string"]
}
Rules:
- Never hallucinate or invent facts.
- If context is partial, confidence <= 0.6.
- Suggest additional files or data that might improve completeness.
"""


# -----------------------------
# Core: Ask a question (multi-domain)
# -----------------------------
def ask_question(question: str, top_k: int = 3) -> RAGAnswer:
    """Retrieve from multiple Chroma domain collections and ask the model."""
    stores = load_all_vectorstores()
    if not stores:
        raise ValueError("No vectorstores found. Run ingestion first!")

    all_results = []
    # Retrieve top-k results per domain (with scores)
    for domain, store in stores:
        try:
            docs_with_scores = store.similarity_search_with_score(question, k=top_k)
            for doc, score in docs_with_scores:
                doc.metadata["domain"] = domain
                doc.metadata["score"] = round(float(score), 3)
                all_results.append(doc)
        except Exception as e:
            print(f"[WARN] Retrieval failed for domain '{domain}': {e}")

    if not all_results:
        return RAGAnswer(
            answer="No relevant information found.",
            confidence=0.0,
            missing_info=["No matching content found in the uploaded documents."],
            citations=[],
            reasoning_summary="No context retrieved.",
            suggestions=["Upload more relevant documents."],
        )

    # -----------------------------
    # Prepare combined context (deduplicated)
    # -----------------------------
    context_blocks = []
    citations = []
    seen_files = set()  # prevent duplicates

    for doc in all_results:
        meta = doc.metadata
        filename = meta.get("filename", "unknown")

        # Skip duplicate entries from the same file
        if filename in seen_files:
            continue
        seen_files.add(filename)

        snippet = doc.page_content[:300].replace("\n", " ")

        citations.append(
            Source(
                doc_id=meta.get("doc_id", ""),
                filename=filename,
                page=meta.get("page", 1),
                score=meta.get("score", 0.0),
                snippet=snippet,
            )
        )

        # attach domain (optional)
        citations[-1].domain = meta.get("domain", "unknown")

        context_blocks.append(
            f"[{filename} | {meta.get('domain', 'unknown')} | score={meta.get('score', 0.0)}] → {snippet}"
        )

    context_text = "\n\n".join(context_blocks)
    user_prompt = f"Question: {question}\n\nContext:\n{context_text}"

    # -----------------------------
    # Call the LLM
    # -----------------------------
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages)

    # -----------------------------
    # Parse JSON output safely
    # -----------------------------
    try:
        data = json.loads(response.content)
    except Exception as e:
        print(f"[WARN] JSON parse error: {e}")
        print("Raw response:", response.content)
        data = {
            "answer": response.content.strip(),
            "confidence": 0.5,
            "missing_info": [],
            "citations": [],
            "reasoning_summary": "Partial fallback response.",
            "suggestions": [],
        }

    # Auto-enrichment if missing info is found
    auto_data = []
    if data.get("missing_info"):
        print("[INFO] Running auto-enrichment for missing topics...")
        auto_data = auto_enrich(data.get("missing_info", []))

    # Append enrichment results to suggestions
    if auto_data:
        data["suggestions"].extend(auto_data)

    # -----------------------------
    # Add domain coverage note (optional)
    # -----------------------------
    retrieved_domains = {c.domain for c in citations if hasattr(c, "domain")}
    all_domains = {"finance", "hr", "sustainability", "general"}
    missing_domains = all_domains - retrieved_domains
    if missing_domains:
        data["reasoning_summary"] += (
            f" No documents found for domain(s): {', '.join(missing_domains)}."
        )

    return RAGAnswer(
        answer=data.get("answer", ""),
        confidence=float(data.get("confidence", 0.5)),
        missing_info=data.get("missing_info", []),
        citations=citations,
        reasoning_summary=data.get("reasoning_summary", ""),
        suggestions=data.get("suggestions", []),
    )


# -----------------------------
# Auto-Enrichment
# -----------------------------
def auto_enrich(missing_topics: List[str]) -> List[str]:
    """Fetch short summaries from DuckDuckGo for missing topics."""
    suggestions = []
    if not missing_topics:
        return suggestions

    with DDGS() as ddgs:
        for topic in missing_topics:
            try:
                results = list(ddgs.text(topic, max_results=1))
                if results:
                    summary = results[0]["body"][:200]
                    suggestions.append(f"Auto-enriched info for '{topic}': {summary}")
            except Exception as e:
                print(f"[WARN] Could not enrich topic '{topic}': {e}")
    return suggestions


# -----------------------------
# Local test
# -----------------------------
if __name__ == "__main__":
    q = "Summarize key financial, HR, and sustainability insights from all documents."
    res = ask_question(q)
    print(json.dumps(res.dict(), indent=2))
