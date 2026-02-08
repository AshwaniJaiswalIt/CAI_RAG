# Hybrid Retrieval-Augmented Generation (Hybrid RAG)

This repository implements a Hybrid RAG system combining dense (sentence-transformers + FAISS) and sparse (BM25) retrieval with Reciprocal Rank Fusion (RRF). It also contains an automated evaluation pipeline (MRR URL-level + two custom metrics) and a minimal Streamlit app.

Contents
- `scripts/` : Core scripts (data collection, preprocessing, index building, retrieval, generation, evaluation)
- `app/` : Minimal Streamlit app to demo query -> answer
- `requirements.txt` : Python dependencies

Quick start (recommended in a virtual environment):

1. Install dependencies

```bash
pip install -r "./requirements.txt"
```

2. Generate fixed URLs (200) and build corpus

```bash
python scripts/fixed_urls_generator.py --out fixed_urls.json
python scripts/data_collection.py --fixed fixed_urls.json --out corpus.json --random 300
```

3. Preprocess and chunk

```bash
python scripts/preprocess.py --in corpus.json --out chunks.json
```

4. Build indices

```bash
python scripts/build_index.py --chunks chunks.json --out_dir indices
```

5. Run evaluation pipeline (generates 100 questions, runs RAG, computes metrics)

```bash
python scripts/evaluate.py --indices indices --chunks chunks.json --questions_out questions.json --report_out report.html
```

6. Start Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

Notes
- The scripts are written to be modular: you can replace embedding or generation models via CLI flags.
- See each script for additional options and parameters.

Report contents, metric definitions, and guidance are implemented in `scripts/evaluate.py` and documented inline.
