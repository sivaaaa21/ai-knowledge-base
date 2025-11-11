# 🧠 AI Knowledge Base Search & Enrichment  
*A lightweight RAG system with document upload, semantic search, and contextual reasoning.*

---

## 🚀 Overview  
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **FastAPI** (backend) and **Streamlit** (frontend).  
It allows users to upload text or PDF files, semantically index them into a **Chroma vector database**, and query the knowledge base using natural language.  
Each answer is supported by **citations**, **confidence scores**, and **domain-specific retrieval**, with **DuckDuckGo auto-enrichment** for missing information.

---

## ⚙️ Tech Stack
- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **Vector Store:** ChromaDB  
- **Embeddings:** HuggingFace MiniLM-L6-v2  
- **LLM:** GPT-4o-mini (OpenAI)  
- **Enrichment:** DuckDuckGo Search (ddgs)

---

## 🧩 Features
- Upload and embed TXT/PDF documents into Chroma collections.  
- Ask natural-language questions across all domains.  
- Retrieve contextual answers with reasoning and confidence.  
- Display citations with similarity scores and snippets.  
- Auto-enrich incomplete responses via DuckDuckGo summaries.

---

## 🧰 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone git@github.com:sivaaaa21/ai-knowledge-base.git
cd ai-knowledge-base
