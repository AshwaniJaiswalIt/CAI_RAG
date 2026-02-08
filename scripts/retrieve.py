"""retrieve.py
Dense retrieval (FAISS), sparse retrieval (BM25), and RRF fusion.
Functions:
- dense_search(query, top_k)
- sparse_search(query, top_k)
- rrf_fuse(list_of_ranked_lists, k=60, top_n=10)
"""
import os
import joblib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re

MODEL_NAME = 'all-MiniLM-L6-v2'

class Retriever:
    def __init__(self, index_dir='indices'):
        self.index_dir = index_dir
        self.index = faiss.read_index(os.path.join(index_dir, 'faiss_index.index'))
        self.meta = joblib.load(os.path.join(index_dir, 'meta.joblib'))
        self.bm25 = joblib.load(os.path.join(index_dir, 'bm25.joblib'))
        self.model = SentenceTransformer(MODEL_NAME)
        self.ids = self.meta['ids']
        self.chunks = self.meta['chunks']

    def dense_search(self, query, top_k=10):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for rank, idx in enumerate(I[0]):
            chunk_id = self.ids[idx]
            score = float(D[0][rank])
            results.append({'chunk_id': chunk_id, 'score': score, 'rank': rank+1})
        return results

    def sparse_search(self, query, top_k=10):
        # Minimal token normalization to match build_index.py: lowercase and remove punctuation
        def normalize(text):
            if not isinstance(text, str):
                return ''
            txt = text.lower()
            txt = re.sub(r"[^\w\s]", " ", txt)
            txt = " ".join(txt.split())
            return txt

        tokens = normalize(query).split()
        # bm25.get_scores returns a list of scores aligned with the corpus order
        scores = np.array(self.bm25.get_scores(tokens))
        if scores.size == 0:
            return []
        # get top_k indices (descending)
        topk_idx = np.argsort(scores)[-top_k:][::-1]
        results = []
        rank = 1
        for idx in topk_idx:
            chunk = self.chunks[idx]
            results.append({'chunk_id': chunk['chunk_id'], 'score': float(scores[idx]), 'rank': rank})
            rank += 1
        return results

    def rrf_fuse(self, dense_list, sparse_list, rrf_k=60, top_n=10):
        # dense_list and sparse_list are lists of dicts with chunk_id and rank
        scores = {}
        for r in dense_list:
            rid = r['chunk_id']
            rank = r['rank']
            scores[rid] = scores.get(rid, 0.0) + 1.0/(rrf_k + rank)
        for r in sparse_list:
            rid = r['chunk_id']
            rank = r['rank']
            scores[rid] = scores.get(rid, 0.0) + 1.0/(rrf_k + rank)
        # sort
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused = []
        for i, (cid, sc) in enumerate(sorted_items[:top_n]):
            chunk = next((c for c in self.chunks if c['chunk_id']==cid), None)
            fused.append({'chunk_id': cid, 'score': sc, 'rank': i+1, 'text': chunk['text'] if chunk else '' , 'url': chunk['url'] if chunk else ''})
        return fused

if __name__ == '__main__':
    # example usage
    r = Retriever('indices')
    q = 'What is the main idea of natural language processing?'
    d = r.dense_search(q, top_k=20)
    s = r.sparse_search(q, top_k=20)
    fused = r.rrf_fuse(d, s, rrf_k=60, top_n=10)
    print('Fused top:', fused[:3])
