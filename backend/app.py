from fastapi import FastAPI, UploadFile, File, Body
from pydantic import BaseModel
from typing import List
import os
import json

from ingestion import ingest_files
from rag_pipeline import ask_question
from schemas import RAGAnswer

app = FastAPI(
    title="AI Knowledge Base Search & Enrichment API",
    description="Upload documents, search them in natural language, and get structured AI-generated answers.",
    version="1.1.0"
)

# -------------------------------
# Request Schema
# -------------------------------
class AskPayload(BaseModel):
    question: str


# -------------------------------
# Health Check
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "message": "API is healthy"}


# -------------------------------
# File Upload Endpoint
# -------------------------------
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """
    Accepts uploaded files (PDF/TXT), saves them to data/uploads/,
    and indexes them into the Chroma vector database.
    """
    upload_dir = os.path.join("data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    saved_files = []
    for f in files:
        file_path = os.path.join(upload_dir, f.filename)
        with open(file_path, "wb") as out:
            out.write(await f.read())
        saved_files.append(file_path)

    count = ingest_files(saved_files)
    return {"status": "indexed", "files": [f.filename for f in files], "chunks_indexed": count}


# -------------------------------
# Question-Answer Endpoint
# -------------------------------
@app.post("/ask", response_model=RAGAnswer)
def ask(payload: AskPayload):
    """
    Takes a question and returns a structured JSON answer
    using the RAG pipeline (retrieval + reasoning).
    """
    return ask_question(payload.question)


# -------------------------------
# Feedback Endpoint (Stretch Goal)
# -------------------------------
@app.post("/feedback")
def feedback(
    rating: int = Body(..., embed=True),
    question: str = Body(..., embed=True),
    comments: str = Body(default="")
):
    """
    Stores user feedback on answer quality.
    - rating: int (1-5)
    - question: original user question
    - comments: optional user notes
    """
    feedback_log = "feedback_log.jsonl"
    feedback_entry = {
        "question": question,
        "rating": rating,
        "comments": comments
    }

    with open(feedback_log, "a") as f:
        json.dump(feedback_entry, f)
        f.write("\n")

    return {"status": "recorded", "message": "Feedback logged successfully"}
