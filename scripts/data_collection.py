"""data_collection.py
Fetch pages from Wikipedia given fixed URLs and sample random URLs for the random set.
Saves raw HTML/text per URL into an output JSON file with fields: url, title, text
"""
import argparse
import json
import random
import time
from wikipediaapi import Wikipedia
from bs4 import BeautifulSoup

wiki = Wikipedia(language='en', user_agent='hybrid-rag-bot/0.1 (contact: you@example.com)')

def fetch_text_from_url(url):
    # wikipedia-api works with titles; extract title from URL
    if not url:
        return None, None
    if '/wiki/' not in url:
        return None, None
    title = url.split('/wiki/')[-1]
    title = title.replace('%20',' ')
    page = wiki.page(title)
    if not page.exists():
        return None, None
    return page.title, page.text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed', required=True, help='Path to fixed_urls.json')
    parser.add_argument('--out', default='corpus.json')
    parser.add_argument('--random', type=int, default=300)
    args = parser.parse_args()

    with open(args.fixed) as f:
        fixed = json.load(f).get('fixed_urls', [])

    results = []
    # Fetch fixed
    for u in fixed:
        title, text = fetch_text_from_url(u)
        if text and len(text.split()) >= 200:
            results.append({'url': u, 'title': title, 'text': text})
        else:
            print(f"Skipping fixed URL (too short or missing): {u}")
        time.sleep(0.1)

    # Sample random pages until we have `args.random` pages
    count = 0
    while count < args.random:
        # use random titles via special pages is not available; instead pick random links from fixed pages
        sample_source = random.choice(results) if results else None
        try:
            if sample_source:
                # pick a link from that page using wikipedia api
                page = wiki.page(sample_source['title'])
                links = list(page.links.keys())
                if not links:
                    continue
                candidate = random.choice(links)
                title = candidate
                p = wiki.page(title)
                if not p.exists():
                    continue
                text = p.text
                url = p.fullurl
            else:
                # fallback: try some common titles
                candidates = ['Earth','Moon','Sun','Albert Einstein','Isaac Newton']
                title = random.choice(candidates)
                p = wiki.page(title)
                text = p.text
                url = p.fullurl
            if text and len(text.split()) >= 200 and url not in [r['url'] for r in results]:
                results.append({'url': url, 'title': p.title, 'text': text})
                count += 1
                print(f"Collected random {count}/{args.random}: {p.title}")
        except Exception as e:
            print('err', e)
        time.sleep(0.1)

    # Save
    with open(args.out, 'w') as f:
        json.dump(results, f)
    print(f"Saved corpus with {len(results)} documents to {args.out}")
