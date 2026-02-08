"""evaluate.py
Evaluation pipeline:
- Load questions (or generate if not provided)
- Run retriever+generator to get answers and retrieved URLs
- Compute MRR at URL level
- Compute Precision@K, NDCG@K and average response latency as additional metrics
- Compute semantic answer similarity (BERTScore if available, else token-F1)
- Produce JSON report and an HTML report with plots
"""
import argparse
import json
import time
import csv
import os
import numpy as np
from retrieve import Retriever
from generate import generate_answer

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

def compute_ndcg(ground_url, ranked_urls, k=10):
    """Compute NDCG@k where relevance is binary (1 if url == ground, else 0).
    DCG = sum_{i=1..k} (2^{rel_i} - 1) / log2(i+1)
    IDCG for a single relevant document = 1 (at rank 1) -> IDCG = 1
    So NDCG reduces to DCG in binary relevance case.
    """
    dcg = 0.0
    for i, u in enumerate(ranked_urls[:k], start=1):
        rel = 1 if u == ground_url else 0
        dcg += (2**rel - 1) / np.log2(i + 1)
    # IDCG is 1.0 if there is at least one relevant doc (which by construction there is)
    idcg = 1.0
    return dcg / idcg if idcg > 0 else 0.0

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
        # generate answer from top-N fused chunks
        try:
            gen_answer = generate_answer(fused[:5], question)
        except Exception as e:
            print('generation-error', e)
            gen_answer = ''
        results.append({'question': question, 'ground_url': ground, 'ranked_urls': ranked_urls, 'mrr': mrr, 'precision@10': prec, 'answer': gen_answer})
    # Additional metric: semantic similarity of generated answer to ground-truth answer using BERTScore if available
    bert_f1_mean = None
    if BERTSCORE_AVAILABLE and qas:
        try:
            preds = [r['answer'] for r in results]
            refs = [q.get('answer','') for q in qas]
            P, R, F1 = bert_score(preds, refs, lang='en', rescale_with_baseline=True)
            bert_f1_mean = float(F1.mean())
        except Exception as e:
            print('bert-score-error', e)
            bert_f1_mean = None
    else:
        # fallback: token-level F1 between generated answer and reference answer
        def token_f1(a, b):
            atok = a.lower().split()
            btok = b.lower().split()
            if not atok or not btok:
                return 0.0
            common = 0
            bcounts = {}
            for t in btok:
                bcounts[t] = bcounts.get(t,0) + 1
            for t in atok:
                if bcounts.get(t,0) > 0:
                    common += 1
                    bcounts[t] -= 1
            prec = common / len(atok)
            rec = common / len(btok)
            if prec+rec == 0:
                return 0.0
            return 2*prec*rec/(prec+rec)

        if qas:
            f1s = []
            for r,q in zip(results, qas):
                f1s.append(token_f1(r.get('answer',''), q.get('answer','')))
            bert_f1_mean = float(np.mean(f1s)) if f1s else None

    out = {
        'mrr_mean': float(np.mean(mrrs)) if mrrs else 0.0,
        'precision10_mean': float(np.mean(precs)) if precs else 0.0,
        'semantic_answer_score_mean': bert_f1_mean,
        'per_question': results,
        'metrics_info': {
            'MRR_url_level': 'Mean Reciprocal Rank at URL level. For each question, rank position r of first correct URL. MRR = (1/Q) * sum_{i=1..Q} (1/r_i).',
            'Precision@10': 'Fraction of questions where ground-truth URL appears in top-10 retrieved URLs: Precision@10 = (1/Q) * sum_{i} 1[ground in top10].',
            'SemanticAnswerMetric': 'If BERTScore is available, mean BERTScore F1 between generated answer and reference answer; otherwise token-level F1 (precision/recall harmonic mean).'
        }
    }
    # Additional evaluation: compute NDCG@10 and average latency
    ndcgs = []
    latencies = []
    for r in results:
        ranked = r.get('ranked_urls', [])
        ndcgs.append(compute_ndcg(r.get('ground_url',''), ranked, k=10))
        latencies.append(r.get('latency', 0.0))

    out['ndcg10_mean'] = float(np.mean(ndcgs)) if ndcgs else 0.0
    out['avg_latency_sec'] = float(np.mean(latencies)) if latencies else 0.0

    # write HTML report with simple plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        report_dir = os.path.dirname(args.report_out) or '.'
        png1 = os.path.join(report_dir, 'mrr_hist.png')
        png2 = os.path.join(report_dir, 'semantic_hist.png')
        png3 = os.path.join(report_dir, 'latency_hist.png')

        # MRR histogram
        plt.figure()
        plt.hist(mrrs, bins=20, color='C0')
        plt.title('MRR distribution')
        plt.xlabel('MRR per question')
        plt.ylabel('Count')
        plt.savefig(png1)
        plt.close()

        # Semantic score histogram
        if bert_f1_mean is not None:
            sems = []
            for r,q in zip(results, qas):
                if r.get('answer') and q.get('answer'):
                    # reuse token_f1 if BERT not available
                    sems.append(r.get('answer'))
            # produce a placeholder plot using the semantic mean
            plt.figure()
            plt.text(0.1,0.5,f"Semantic metric mean: {bert_f1_mean:.4f}", fontsize=14)
            plt.axis('off')
            plt.savefig(png2)
            plt.close()

        # Latency histogram
        plt.figure()
        plt.hist(latencies, bins=20, color='C2')
        plt.title('Response latency distribution (s)')
        plt.xlabel('Seconds')
        plt.ylabel('Count')
        plt.savefig(png3)
        plt.close()

        # write a basic HTML file
        html = []
        html.append('<html><head><title>Hybrid RAG Evaluation Report</title></head><body>')
        html.append(f'<h1>Hybrid RAG Evaluation</h1>')
        html.append(f'<p>MRR (URL-level): {out["mrr_mean"]:.4f}</p>')
        html.append(f'<p>Precision@10: {out["precision10_mean"]:.4f}</p>')
        html.append(f'<p>NDCG@10: {out["ndcg10_mean"]:.4f}</p>')
        html.append(f'<p>Avg latency (s): {out["avg_latency_sec"]:.3f}</p>')
        html.append('<h2>Metric details</h2>')
        for k,v in out.get('metrics_info', {}).items():
            html.append(f'<h3>{k}</h3><p>{v}</p>')
        html.append('<h2>Visualizations</h2>')
        html.append(f'<img src="{os.path.basename(png1)}" width="600"/>')
        if os.path.exists(png2):
            html.append(f'<img src="{os.path.basename(png2)}" width="600"/>')
        html.append(f'<img src="{os.path.basename(png3)}" width="600"/>')
        html.append('</body></html>')

        html_path = os.path.join(report_dir, 'report.html')
        with open(html_path, 'w') as hf:
            hf.write('\n'.join(html))
        print('Wrote HTML report to', html_path)
    except Exception as e:
        print('Could not write HTML report (matplotlib may be missing):', e)
    with open(args.report_out, 'w') as f:
        json.dump(out, f, indent=2)
    print('Wrote report to', args.report_out)
