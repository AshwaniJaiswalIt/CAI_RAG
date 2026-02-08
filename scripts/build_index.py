"""build_index.py
Embeds chunks using sentence-transformers and builds a FAISS index. Also builds BM25 index (rank_bm25).
Saves indices to specified output directory.
"""
import argparse
import json
import os
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import joblib

MODEL_NAME = 'all-MiniLM-L6-v2'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks', required=True)
    parser.add_argument('--out_dir', default='indices')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_chunks', type=int, default=None, help='If set, embed only the first N chunks (useful for smoke tests)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.chunks, 'r') as f:
        chunks = json.load(f)

    if args.max_chunks is not None:
        chunks = chunks[:args.max_chunks]

    texts = [c['text'] for c in chunks]
    ids = [c['chunk_id'] for c in chunks]

    print('Loading model', MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    # Embed
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    # Build FAISS index (cosine similarity via inner product on normalized vectors)
    index = faiss.IndexFlatIP(dim)
    # normalize
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(args.out_dir, 'faiss_index.index'))
    # Save metadata mapping
    joblib.dump({'ids': ids, 'chunks': chunks}, os.path.join(args.out_dir, 'meta.joblib'))

    # Build BM25 with minimal token normalization (lowercase, remove punctuation)
    def normalize(text):
        if not isinstance(text, str):
            return ''
        txt = text.lower()
        txt = re.sub(r"[^\w\s]", " ", txt)
        txt = " ".join(txt.split())
        return txt

    tokenized = [normalize(t).split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    joblib.dump(bm25, os.path.join(args.out_dir, 'bm25.joblib'))
    print('Indices saved to', args.out_dir)
