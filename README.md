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
```

### 2️⃣ Setup Virtual Environment
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Add Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
```

### 5️⃣ Run Backend
```bash
uvicorn backend.app:app --reload
```
Access API at: http://127.0.0.1:8000

### 6️⃣ Run Frontend
```bash
streamlit run frontend/app.py
```
Access UI at: http://localhost:8501
