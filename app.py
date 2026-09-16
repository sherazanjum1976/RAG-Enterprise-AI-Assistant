import os
import json
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # must match Colab indexing notebook
FAISS_INDEX_PATH = "faiss_index"
METADATA_PATH = "metadata.json"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 5

st.set_page_config(page_title="Enterprise Document QA", layout="wide")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"FAISS index not found at '{FAISS_INDEX_PATH}'. Upload it alongside app.py.")
        st.stop()
    return faiss.read_index(FAISS_INDEX_PATH)


@st.cache_resource
def load_metadata():
    if not os.path.exists(METADATA_PATH):
        st.error(f"metadata.json not found at '{METADATA_PATH}'. Upload it alongside app.py.")
        st.stop()
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_groq_client():
    api_key = None
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY is not set. Add it under Streamlit Cloud → Settings → Secrets, "
            "or set it as an environment variable."
        )
        st.stop()
    return Groq(api_key=api_key)


def retrieve_chunks(query, embed_model, index, metadata, top_k=TOP_K):
    if index.ntotal != len(metadata):
        st.warning("FAISS index and metadata.json record counts do not match. Results may be unreliable.")

    query_vector = embed_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    k = min(top_k, index.ntotal)
    if k == 0:
        return []

    scores, indices = index.search(query_vector, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        record = metadata[idx]
        results.append({**record, "score": float(score)})
    return results


def build_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks):
        page_info = f", page {chunk['page_number']}" if chunk.get("page_number") else ""
        parts.append(f"[Source {i + 1}: {chunk['source']}{page_info}]\n{chunk['text']}")
    return "\n\n".join(parts)


def generate_answer(client, question, context):
    if not context.strip():
        return "I could not find any relevant information in the indexed documents to answer this question."

    system_prompt = (
        "You are an enterprise document QA assistant. Answer the user's question using ONLY "
        "the provided context. Do not use outside knowledge and do not make up information. "
        "If the context does not contain enough information to answer, say clearly: "
        "'The answer is not available in the provided documents.' Be concise and factual."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer using only the context above:"

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer from Groq API: {e}"


def main():
    st.title("📄 Enterprise Document QA System")
    st.caption("Ask questions about your indexed documents. Answers are grounded only in retrieved content.")

    embed_model = load_embedding_model()
    index = load_faiss_index()
    metadata = load_metadata()
    client = get_groq_client()

    st.sidebar.header("Index Info")
    st.sidebar.write(f"Indexed chunks: {index.ntotal}")
    st.sidebar.write(f"Metadata records: {len(metadata)}")
    st.sidebar.write(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    st.sidebar.write(f"LLM model: {GROQ_MODEL}")

    question = st.text_input("Ask a question about your documents:")

    if st.button("Get Answer") and question.strip():
        with st.spinner("Retrieving relevant context..."):
            chunks = retrieve_chunks(question, embed_model, index, metadata)

        if not chunks:
            st.warning("No relevant content found in the index.")
            return

        context = build_context(chunks)

        with st.spinner("Generating answer..."):
            answer = generate_answer(client, question, context)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        seen = set()
        for chunk in chunks:
            key = (chunk["filename"], chunk["folder_path"], chunk.get("page_number"))
            if key in seen:
                continue
            seen.add(key)
            page_info = f" | Page: {chunk['page_number']}" if chunk.get("page_number") else ""
            st.markdown(
                f"**{chunk['filename']}** — Folder: `{chunk['folder_path']}`{page_info} "
                f"(relevance: {chunk['score']:.3f})"
            )
            with st.expander("View excerpt"):
                st.write(chunk["text"])


if __name__ == "__main__":
    main()
