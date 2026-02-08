"""preprocess.py
Cleans raw text and chunks documents into 200-400 token chunks with 50-token overlap.
Outputs chunks JSON with metadata: chunk_id, url, title, text, start_word, end_word
"""
import argparse
import json
import hashlib
import uuid
from tqdm import tqdm

def clean(text):
    # basic cleaning
    return ' '.join(text.split())

def chunk_text(text, min_words=200, max_words=400, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    L = len(words)
    while i < L:
        end = i + max_words
        chunk_words = words[i:end]
        if len(chunk_words) < min_words:
            # attach to previous if too small
            if chunks:
                chunks[-1]['text'] += ' ' + ' '.join(chunk_words)
                chunks[-1]['end_word'] = L
            else:
                chunks.append({'text':' '.join(chunk_words), 'start_word':i, 'end_word':L})
            break
        chunks.append({'text': ' '.join(chunk_words), 'start_word': i, 'end_word': end})
        i = end - overlap
    return chunks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', default='chunks.json')
    args = parser.parse_args()

    with open(args.infile, 'r') as f:
        docs = json.load(f)
    chunks_out = []
    for d in tqdm(docs):
        text = clean(d.get('text',''))
        if not text:
            continue
        cks = chunk_text(text, min_words=200, max_words=400, overlap=50)
        for idx, c in enumerate(cks):
            chunk_id = hashlib.sha1((d['url'] + str(idx)).encode()).hexdigest()
            chunks_out.append({
                'chunk_id': chunk_id,
                'url': d['url'],
                'title': d.get('title',''),
                'text': c['text'],
                'start_word': c['start_word'],
                'end_word': c['end_word']
            })
    with open(args.out, 'w') as f:
        json.dump(chunks_out, f)
    print(f"Wrote {len(chunks_out)} chunks to {args.out}")
