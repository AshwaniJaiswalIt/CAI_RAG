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
import requests

# polite user agent
wiki = Wikipedia(language='en', user_agent='hybrid-rag-bot/0.1 (contact: you@example.com)')

RANDOM_PAGE_URL = 'https://en.wikipedia.org/wiki/Special:Random'

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
    parser.add_argument('--fixed', required=True, help='Path to fixed_urls.json (must contain exactly 200 unique URLs)')
    parser.add_argument('--out', default='corpus.json')
    parser.add_argument('--random', type=int, default=300, help='Number of random pages to sample for this run')
    parser.add_argument('--min_words', type=int, default=200, help='Minimum words required per page')
    parser.add_argument('--max_tries', type=int, default=3000, help='Maximum attempts to find random pages')
    args = parser.parse_args()

    # Load fixed URLs. Accept either a list or a dict with key 'fixed_urls'
    with open(args.fixed) as f:
        data = json.load(f)
        if isinstance(data, dict) and 'fixed_urls' in data:
            fixed = data['fixed_urls']
        elif isinstance(data, list):
            fixed = data
        else:
            raise SystemExit(f"Unrecognized fixed URLs format in {args.fixed}; expected list or {{'fixed_urls': [...]}}")

    # Validate fixed set size
    fixed_set = list(dict.fromkeys(fixed))
    if len(fixed_set) != 200:
        raise SystemExit(f"fixed_urls must contain exactly 200 unique URLs. Found {len(fixed_set)}. Please provide a file with 200 unique URLs.")

    results = []
    seen_urls = set()

    # Fetch fixed set (these must all be present and meet min_words)
    for u in fixed_set:
        title, text = fetch_text_from_url(u)
        if text and len(text.split()) >= args.min_words:
            results.append({'url': u, 'title': title, 'text': text})
            seen_urls.add(u)
        else:
            raise SystemExit(f"Fixed URL does not meet minimum word requirement or could not be fetched: {u}")
        time.sleep(0.05)

    # Sample random pages via Special:Random redirect until we have args.random unique pages
    random_count = 0
    tries = 0
    while random_count < args.random and tries < args.max_tries:
        tries += 1
        try:
            # follow redirect to obtain a random page URL
            r = requests.get(RANDOM_PAGE_URL, allow_redirects=True, timeout=10)
            final_url = r.url
            # extract title from final_url
            if '/wiki/' not in final_url:
                continue
            title = final_url.split('/wiki/')[-1]
            # use wikipediaapi to fetch page content
            p = wiki.page(title)
            if not p.exists():
                continue
            text = p.text
            url = p.fullurl
            if not text or len(text.split()) < args.min_words:
                continue
            if url in seen_urls:
                continue
            results.append({'url': url, 'title': p.title, 'text': text})
            seen_urls.add(url)
            random_count += 1
            if random_count % 10 == 0 or random_count <= 5:
                print(f"Collected random {random_count}/{args.random}: {p.title}")
        except Exception as e:
            print('random-sample-error', e)
        time.sleep(0.05)

    if random_count < args.random:
        raise SystemExit(f"Failed to collect {args.random} random pages within {args.max_tries} tries (collected {random_count}). Try increasing --max_tries or run in a runtime with network access.")

    # Save combined corpus (fixed + random)
    with open(args.out, 'w') as f:
        json.dump(results, f)
    print(f"Saved corpus with {len(results)} documents to {args.out} (fixed={len(fixed_set)}, random={random_count})")
