import os
import uuid
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader



# -------------------------
# Helper: Extract text from file
# -------------------------
def extract_text(file_path: str) -> str:
    """Extracts text from PDF or TXT files."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    if ext == ".pdf":
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path}: {e}")

    elif ext == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path}: {e}")
    else:
        print(f"[WARN] Unsupported file type: {file_path}")

    return text


# -------------------------
# Helper: Split text into chunks
# -------------------------
def split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    """Splits long text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


# -------------------------
# Main: Ingest files into Chroma
# -------------------------
def ingest_files(files: List[str], category: str = "general"):
    """
    Takes a list of files, extracts text, splits into chunks,
    and saves to a ChromaDB collection.
    """
    # Initialize vector store path
    persist_directory = os.path.join(os.getcwd(), "data", "chroma")
    os.makedirs(persist_directory, exist_ok=True)

    # Initialize embeddings (OpenAI by default)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create or load a Chroma collection
    vectorstore = Chroma(
        collection_name=f"docs_{category}",   # <-- dynamically use the domain name
        embedding_function=embeddings,
        persist_directory=persist_directory
    )


    total_chunks = 0

    for file_path in files:
        print(f"[INFO] Processing: {file_path}")
        text = extract_text(file_path)

        if not text.strip():
            print(f"[WARN] No text extracted from {file_path}, skipping.")
            continue

        chunks = split_text(text)
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadatas.append({
                "filename": os.path.basename(file_path),
                "chunk_id": i,
                "doc_id": str(uuid.uuid4())
            })

        # Add to Chroma
        vectorstore.add_texts(
            texts=chunks,
            metadatas=metadatas,
            ids=[m["doc_id"] for m in metadatas]
        )

        total_chunks += len(chunks)
        print(f"[INFO] Added {len(chunks)} chunks from {os.path.basename(file_path)}")

    # Persist for reuse
    # vectorstore.persist()
    print(f"[DONE] Indexed {total_chunks} chunks total.")
    return total_chunks

if __name__ == "__main__":
    # Ingest domain-based folders automatically
    base_dir = os.path.join("data", "uploads")
    os.makedirs(base_dir, exist_ok=True)

    categories = ["finance", "hr", "sustainability", "general"]
    total_chunks = 0

    for category in categories:
        folder = os.path.join(base_dir, category)
        os.makedirs(folder, exist_ok=True)

        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".pdf", ".txt"))
        ]

        if not files:
            print(f"[INFO] No files found in {folder}, skipping...")
            continue

        print(f"[INFO] Found {len(files)} files in '{category}', starting ingestion...")
        count = ingest_files(files, category=category)
        total_chunks += count

    print(f"[DONE] Indexed {total_chunks} chunks total across all categories.")

