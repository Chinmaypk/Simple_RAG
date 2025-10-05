import os
import glob
import pickle
import re
from pathlib import Path

import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --------------------------
# CONFIG
# --------------------------
DATA_DIR = "sample_docs"
INDEX_PATH = "vector.index"
CHUNKS_PATH = "chunks.pkl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
KEY_FILE = "gemini_key.txt"

# --------------------------
# GLOBALS
# --------------------------
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
index = None
chunks = []  # list of dicts: {"text":..., "source":..., "type":..., "label":...}

# --------------------------
# HELPERS
# --------------------------

def get_api_key():
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        st.error(f"API key file '{KEY_FILE}' not found!")
        return None

def load_documents(data_dir):
    docs = []
    for ext in ["*.txt", "*.md", "*.py", "*.js", "*.ts", "*.cpp", "*.java"]:
        for file_path in glob.glob(os.path.join(data_dir, ext)):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                docs.append((Path(file_path).name, text))
    return docs

def chunk_text(text, chunk_size=500, overlap=50):
    """Chunk normal text into overlapping windows."""
    result = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        result.append(text[start:end])
        start += chunk_size - overlap
    return result

def chunk_code(text):
    """
    Chunk code by function/class or approx 20 lines per chunk
    to keep semantic boundaries.
    """
    pattern = re.compile(r"(^def .+?:|^class .+?:)", re.MULTILINE)
    chunks = []
    current_chunk = []

    lines = text.split("\n")
    for line in lines:
        if pattern.match(line) and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
        else:
            current_chunk.append(line)

        if len(current_chunk) > 20:
            chunks.append("\n".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def build_index():
    """Build and save FAISS index and chunk metadata."""
    global index, chunks
    docs = load_documents(DATA_DIR)
    chunks = []
    vectors = []

    for fname, doc in docs:
        if fname.endswith((".py", ".js", ".ts", ".cpp", ".java")):
            doc_chunks = chunk_code(doc)
            chunk_type = "code"
        else:
            doc_chunks = chunk_text(doc)
            chunk_type = "text"

        for c in doc_chunks:
            chunks.append({"text": c, "source": fname, "type": chunk_type})
            vec = embed_model.encode(c)
            vectors.append(vec)

    vectors = np.vstack(vectors).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # Save index and chunks
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

def load_index():
    """Load FAISS index and chunk metadata if available."""
    global index, chunks
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return True
    return False

def retrieve(query, k=TOP_K):
    if index is None:
        st.warning("Please build the index first!")
        return []

    q_vec = embed_model.encode(query).astype("float32").reshape(1, -1)
    distances, idxs = index.search(q_vec, k)
    results = []
    for i, dist in zip(idxs[0], distances[0]):
        if i == -1:
            continue
        results.append({**chunks[i], "score": float(dist)})
    return results

def generate_answer_with_gemini(query, retrieved):
    api_key = get_api_key()
    if not api_key:
        return "No API key found.", ""

    genai.configure(api_key=api_key)

    context = ""
    for i, r in enumerate(retrieved, 1):
        context += f"Source [{i}] ({r['source']}):\n{r['text']}\n\n"

    prompt = f"""
You are a helpful assistant for both text and code.
Answer the question based ONLY on the provided context.
If you show code, present it verbatim inside fenced code blocks.
Be concise (max 150 words). If you don't know, say "I don't know".
Use inline citations like [1], [2] referencing the sources.

Context:
{context}

Question: {query}
Answer:
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content(prompt)
    return resp.text.strip()

def extract_cited_indices(answer_text):
    return set(int(num) for num in re.findall(r"\[(\d+)\]", answer_text))

# --------------------------
# STREAMLIT UI
# --------------------------
st.set_page_config(page_title="Code & Text RAG with Gemini", page_icon="💻")
st.title("💻 Simple Code + Text RAG Demo (Gemini)")

# Load saved index if available
if load_index():
    st.sidebar.success("✅ Index loaded from disk")
else:
    st.sidebar.warning("⚠️ No saved index found, please build it.")

if st.sidebar.button("📥 Build / Rebuild Index"):
    with st.spinner("Building index..."):
        build_index()
    st.sidebar.success("Index built and saved!")

user_query = st.text_input("Ask a question (about text or code):")

if user_query:
    with st.spinner("Retrieving relevant chunks..."):
        retrieved = retrieve(user_query, k=TOP_K)

    with st.spinner("Generating answer ..."):
        answer = generate_answer_with_gemini(user_query, retrieved)

    st.subheader("💬 Answer")
    st.write(answer)

    cited_indices = extract_cited_indices(answer)
    st.markdown("### 📚 Sources Used")
    if cited_indices:
        for i, r in enumerate(retrieved, 1):
            if i in cited_indices:
                with st.expander(f"[{i}] {r['source']} ({r['type']})"):
                    if r['type'] == "code":
                        st.code(r['text'], language="python")
                    else:
                        st.write(r['text'])
    else:
        st.info("No specific sources were cited by the model.")
