"""Minimal Streamlit app demonstrating the Hybrid RAG retrieval and generation.
Run with: streamlit run app/streamlit_app.py
"""
import streamlit as st
import time
from retrieve import Retriever
from generate import generate_answer

st.title('Hybrid RAG Demo')
retriever = Retriever('indices')

q = st.text_input('Enter your question')
if st.button('Run') and q:
    start = time.time()
    dense = retriever.dense_search(q, top_k=50)
    sparse = retriever.sparse_search(q, top_k=50)
    fused = retriever.rrf_fuse(dense, sparse, rrf_k=60, top_n=10)
    answer = generate_answer(fused[:5], q)
    elapsed = time.time() - start
    st.subheader('Answer')
    st.write(answer)
    st.subheader('Retrieved chunks (top 10)')
    for i, f in enumerate(fused):
        st.write(f"{i+1}. URL: {f['url']}  | RRF score: {f['score']:.4f}")
        st.write(f"{f['text'][:400]}...")
    st.write(f"Response time: {elapsed:.2f}s")
