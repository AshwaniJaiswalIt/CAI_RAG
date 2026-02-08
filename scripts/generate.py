"""generate.py
Helper to generate an answer from retrieved context using a seq2seq model (e.g., flan-t5-base).
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = 'google/flan-t5-base'

def generate_answer(context_chunks, question, max_input_tokens=1024, max_answer_tokens=256):
    # concatenate top-N chunks with separators
    text = '\n\n'.join([c['text'] for c in context_chunks])
    prompt = f"Context: {text}\n\nQuestion: {question}\nAnswer:"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_input_tokens)
    outputs = model.generate(**inputs, max_length=max_answer_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == '__main__':
    # quick demo
    ctx = [{'text': 'Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language.'}]
    print(generate_answer(ctx, 'What is NLP?'))
