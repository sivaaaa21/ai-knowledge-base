import streamlit as st
import requests
import json

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
API_BASE = "http://127.0.0.1:8000"  # FastAPI backend
st.set_page_config(page_title="AI Knowledge Search", layout="wide")

# ------------------------------------------------
# HEADER
# ------------------------------------------------
st.title("📚 AI Knowledge Base Search & Enrichment")
st.markdown("Upload documents → Ask questions → Get contextual AI answers")

# ------------------------------------------------
# UPLOAD SECTION
# ------------------------------------------------
st.header("📤 Upload Documents")
uploaded_files = st.file_uploader(
    "Upload one or more PDF or TXT files", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

if st.button("Index Documents") and uploaded_files:
    files = [("files", (f.name, f, f.type)) for f in uploaded_files]
    with st.spinner("Uploading and indexing..."):
        res = requests.post(f"{API_BASE}/upload", files=files)
    if res.status_code == 200:
        st.success(f"✅ Indexed {res.json().get('chunks_indexed', 0)} chunks.")
    else:
        st.error(f"❌ Upload failed: {res.text}")

# ------------------------------------------------
# QUESTION SECTION
# ------------------------------------------------
st.header("❓ Ask a Question")
question = st.text_area("Type your question below:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            payload = {"question": question}
            res = requests.post(f"{API_BASE}/ask", json=payload)

        if res.status_code != 200:
            st.error(f"Backend error: {res.text}")
        else:
            data = res.json()

            # -------------------------------
            # DISPLAY ANSWER SECTION
            # -------------------------------
            st.subheader("💬 Answer")
            st.write(data["answer"])
            st.caption(f"Confidence: {data['confidence']:.2f}")

            # -------------------------------
            # DISPLAY MISSING INFO
            # -------------------------------
            if data.get("missing_info"):
                st.warning("⚠️ Missing Information:")
                for item in data["missing_info"]:
                    st.write(f"- {item}")

            # -------------------------------
            # DISPLAY SUGGESTIONS
            # -------------------------------
            if data.get("suggestions"):
                st.info("💡 Suggestions / Enrichments:")
                for s in data["suggestions"]:
                    st.write(f"- {s}")

            # -------------------------------
            # DISPLAY CITATIONS (with scores)
            # -------------------------------
            if data.get("citations"):
                st.subheader("📄 Citations")

                for c in data["citations"]:
                    filename = c.get("filename", "unknown")
                    page = c.get("page", "N/A")
                    score = c.get("score", 0.0)
                    domain = c.get("domain", "unknown")
                    snippet = c.get("snippet", "")

                    # Display file name, score, and domain
                    label = f"📘 {filename} | 🏷️ Domain: {domain} | 🎯 Score: {score:.3f}"
                    with st.expander(label):
                        st.markdown(f"""
                        **Snippet (Page {page}):**
                        > {snippet}

                        **Score:** `{score:.3f}`  
                        **Domain:** `{domain}`
                        """)
