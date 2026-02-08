"""generate_questions.py
Generate question-answer pairs from the corpus. This script uses a T5-based question generation model to create Qs from text.
Note: quality depends on available models; you can replace model names as needed.
"""
import argparse
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

MODEL = 'valhalla/t5-small-qg-hl'

# Simple helper: highlight answer spans by selecting top sentences.

def split_into_sentences(text, max_len=400):
    sents = text.split('.')
    return [s.strip() for s in sents if len(s.strip())>20]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks', required=True, help='Chunks JSON')
    parser.add_argument('--out', default='questions.json')
    parser.add_argument('--num_questions', type=int, default=100)
    args = parser.parse_args()

    with open(args.chunks) as f:
        chunks = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

    qas = []
    for c in tqdm(chunks):
        sents = split_into_sentences(c['text'])
        if not sents:
            continue
        # pick one sentence as answer
        ans = sents[0]
        # create input in the expected format: "generate question: <context> <hl> <answer> <hl>"
        input_text = f"generate question: {c['text']} <hl> {ans} <hl>"
        inputs = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=512)
        outputs = model.generate(inputs, max_length=64)
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        qas.append({'question': question, 'answer': ans, 'url': c['url'], 'chunk_id': c['chunk_id']})
        if len(qas) >= args.num_questions:
            break
    with open(args.out, 'w') as f:
        json.dump(qas, f, indent=2)
    print(f'Wrote {len(qas)} Q&A pairs to {args.out}')
