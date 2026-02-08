"""fixed_urls_generator.py
Generates a fixed list of Wikipedia URLs (200) that have at least 200 words.

Outputs a JSON file with a list of URLs.
"""
import argparse
import json
import random
import time
from wikipediaapi import Wikipedia
from bs4 import BeautifulSoup

wiki_wiki = Wikipedia(language='en', user_agent='hybrid-rag-bot/0.1 (contact: you@example.com)')

def page_word_count(title):
    p = wiki_wiki.page(title)
    if not p.exists():
        return 0, None
    # strip references from summary by using the extract
    text = p.text or ""
    words = len(text.split())
    return words, p.fullurl

def generate(n=200, start_category=None):
    # Strategy: use random popular titles (random sampling of 'Special:Random' isn't available in wikipedia-api),
    # fallback: use a curated set of seed titles and expand via links.
    seeds = [
        'Python (programming language)', 'Machine learning', 'Artificial intelligence',
        'Natural language processing', 'Economics', 'History of the United States',
        'World War II', 'Physics', 'Biology', 'Chemistry', 'Mathematics', 'Geography',
        'Philosophy', 'Sociology', 'Psychology', 'Music', 'Film', 'Literature', 'Environmentalism'
    ]
    urls = []
    seen = set()
    titles_to_try = seeds[:]
    idx = 0
    while len(urls) < n and idx < len(titles_to_try):
        title = titles_to_try[idx]
        idx += 1
        words, url = page_word_count(title)
        if words >= 200 and url and url not in seen:
            urls.append(url)
            seen.add(url)
        # expand links
        p = wiki_wiki.page(title)
        for k in list(p.links.keys())[:50]:
            if k not in seen and k not in titles_to_try:
                titles_to_try.append(k)
        time.sleep(0.1)
        if idx % 50 == 0:
            print(f"Tried {idx} titles, collected {len(urls)} URLs so far")
    # If still not enough, sample from 'List of country' pages and their links
    if len(urls) < n:
        # add some high-probability pages by enumerating well-known titles
        extra_titles = [
            'United States', 'United Kingdom', 'India', 'China', 'Canada', 'Australia',
            'Germany', 'France', 'Italy', 'Spain', 'Brazil', 'Russia', 'Japan'
        ]
        for t in extra_titles:
            words, url = page_word_count(t)
            if words >= 200 and url not in seen:
                urls.append(url); seen.add(url)
            if len(urls) >= n:
                break
    if len(urls) < n:
        print(f"Warning: only collected {len(urls)} URLs; increase seed list or relax constraints")
    return urls[:n]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='fixed_urls.json')
    parser.add_argument('--n', type=int, default=200)
    args = parser.parse_args()
    urls = generate(n=args.n)
    with open(args.out, 'w') as f:
        json.dump({'fixed_urls': urls}, f, indent=2)
    print(f"Wrote {len(urls)} URLs to {args.out}")
