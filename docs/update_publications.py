#!/usr/bin/env python3
"""
Fetch the publications that mention or cite pySDC from the Helmholtz Research Software Directory
(https://helmholtz.software/software/pysdc) into docs/source/publications.json, which the website's
publications page is rendered from.

The directory lists the reference papers of pySDC (the ACM TOMS paper and the Zenodo record), the publications
its maintainers added as mentions, and the publications that cite a reference paper, which it finds itself. To
add a publication to the page, add it as a mention there.

If the directory cannot be reached, the committed file is left alone, so that the website still builds.
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

API = 'https://helmholtz.software/api/v1'
SOFTWARE = '11b0ba71-a474-4bde-8b06-6e878b968f55'  # pySDC in the directory
FIELDS = 'id,doi,url,title,authors,publisher,journal,publication_year,mention_type'
OUT = Path(__file__).parent / 'source' / 'publications.json'


def get(query):
    with urllib.request.urlopen(f'{API}/{query}', timeout=30) as response:
        return json.load(response)


def main():
    try:
        references = [
            m['mention'] for m in get(f'reference_paper_for_software?software=eq.{SOFTWARE}&select=mention({FIELDS})')
        ]
        mentions = [m['mention'] for m in get(f'mention_for_software?software=eq.{SOFTWARE}&select=mention({FIELDS})')]
        ids = ','.join(reference['id'] for reference in references)
        citations = [c['citation'] for c in get(f'citation_for_mention?mention=in.({ids})&select=citation({FIELDS})')]
    except OSError as error:
        print(f'Could not fetch the publications, keeping {OUT.name}: {error}', file=sys.stderr)
        return

    reference_dois = {reference['doi'].lower() for reference in references if reference['doi']}
    publications = {}
    for publication in mentions + citations:
        if (publication['doi'] or '').lower() in reference_dois or not publication['title']:
            continue
        # the same work can be both a mention and a citation, or come as a preprint and as an article
        key = re.sub(r'\W+', '', publication['title'].lower())
        known = publications.get(key)
        if known is None or (
            known['mention_type'] != 'journalArticle' and publication['mention_type'] == 'journalArticle'
        ):
            publications[key] = publication

    entries = [
        {
            'title': p['title'],
            'authors': p['authors'],
            'year': p['publication_year'],
            'venue': p['journal'] or p['publisher'],
            'doi': p['doi'],
            'url': p['url'],
        }
        for p in publications.values()
    ]
    entries.sort(key=lambda entry: (-(entry['year'] or 0), entry['title'].lower()))
    OUT.write_text(json.dumps(entries, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {len(entries)} publications to {OUT}')


if __name__ == '__main__':
    main()
