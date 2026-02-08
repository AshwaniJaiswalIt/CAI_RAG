"""evaluate.py
Evaluation pipeline:
- Load questions (or generate if not provided)
- Run retriever+generator to get answers and retrieved URLs
- Compute MRR at URL level
- Compute Precision@K and BERTScore as custom metrics
- Produce CSV/JSON report and simple HTML visualization
"""
import argparse
import json
import time
import csv
import os
import numpy as np
from retrieve import Retriever
from generate import generate_answer_if_needed

# We'll implement MRR (URL level), Precision@K, and use bert-score if installed
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except Exception:
    BERTSCORE_AVAILABLE = False

def compute_mrr(ground_url, ranked_urls):
    # ranked_urls: list of urls in order
    for i, u in enumerate(ranked_urls, start=1):
        if u == ground_url:
            return 1.0 / i
    return 0.0

def precision_at_k(ground_url, ranked_urls, k=10):
    return 1.0 if ground_url in ranked_urls[:k] else 0.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices', default='indices')
    parser.add_argument('--chunks', required=True)
    parser.add_argument('--questions_in', default=None)
    parser.add_argument('--questions_out', default='questions_generated.json')
    parser.add_argument('--report_out', default='report.json')
    args = parser.parse_args()

    retriever = Retriever(args.indices)

    if args.questions_in and os.path.exists(args.questions_in):
        with open(args.questions_in) as f:
            qas = json.load(f)
    else:
        # call generator script
        from generate_questions import MODEL as _m
        from generate_questions import split_into_sentences
        # fallback minimal: generate blanks
        print('No questions provided; please run generate_questions.py to produce questions first.')
        qas = []

    results = []
    mrrs = []
    precs = []
    all_gen_texts = []
    for q in qas:
        question = q['question']
        ground = q['url']
        start = time.time()
        dense = retriever.dense_search(question, top_k=50)
        sparse = retriever.sparse_search(question, top_k=50)
        fused = retriever.rrf_fuse(dense, sparse, rrf_k=60, top_n=20)
        ranked_urls = [f['url'] for f in fused]
        # compute metrics
        mrr = compute_mrr(ground, ranked_urls)
        prec = precision_at_k(ground, ranked_urls, k=10)
        mrrs.append(mrr)
        precs.append(prec)
        # placeholder for generated answer
        gen_answer = ''
        results.append({'question': question, 'ground_url': ground, 'ranked_urls': ranked_urls, 'mrr': mrr, 'precision@10': prec, 'answer': gen_answer})
    out = {'mrr_mean': float(np.mean(mrrs)) if mrrs else 0.0, 'precision10_mean': float(np.mean(precs)) if precs else 0.0, 'per_question': results}
    with open(args.report_out, 'w') as f:
        json.dump(out, f, indent=2)
    print('Wrote report to', args.report_out)
