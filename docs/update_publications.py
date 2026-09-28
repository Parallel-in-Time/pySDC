#!/usr/bin/env python3
"""
Fetch the publications that mention or cite pySDC from the Helmholtz Research Software Directory
(https://helmholtz.software/software/pysdc) into docs/source/publications.json, which the website's
publications page is rendered from.

The directory lists the reference papers of pySDC (the ACM TOMS paper and the Zenodo record), the publications
its maintainers added as mentions, and the publications that cite a reference paper, which it finds itself. To
add a publication to the page, add it as a mention there.

It also fetches the metadata of the papers the project gallery names for each project (`papers` in
pySDC/projects/gallery.yml) from Crossref, or DataCite for arXiv DOIs, into docs/source/project_papers.json, which
each project's page lists them from.

If a source cannot be reached, the committed file is left alone, so that the website still builds.
"""

import html
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = 'https://helmholtz.software/api/v1'
SOFTWARE = '11b0ba71-a474-4bde-8b06-6e878b968f55'  # pySDC in the directory
FIELDS = 'id,doi,url,title,authors,publisher,journal,publication_year,mention_type'
OUT = Path(__file__).parent / 'source' / 'publications.json'
GALLERY = Path(__file__).parents[1] / 'pySDC' / 'projects' / 'gallery.yml'
PAPERS_OUT = Path(__file__).parent / 'source' / 'project_papers.json'

# Corrections to the directory's data, until they are made there (by DOI or, without one, by URL). None drops an
# entry: e.g. a preprint whose published version is listed as well.
CORRECTIONS = {
    'http://arxiv.org/pdf/2103.12571.pdf': None,  # the preprint of 10.2140/camcos.2023.18.55
    '10.48550/arxiv.2002.07555': {  # the preprint; this is the published version
        'title': 'Convergence of multilevel spectral deferred corrections',
        'year': 2021,
        'venue': 'Communications in Applied Mathematics and Computational Science',
        'doi': '10.2140/camcos.2021.16.227',
        'url': 'https://doi.org/10.2140/camcos.2021.16.227',
    },
}

# Corrections to the metadata of the project papers, by DOI
PAPER_CORRECTIONS = {
    '10.34734/fzj-2026-03360': {'type': 'phdthesis', 'venue': 'Technische Universität Hamburg'},  # DataCite: a book
}


def get(query):
    with urllib.request.urlopen(f'{API}/{query}', timeout=30) as response:
        return json.load(response)


def complete_title(title, doi):
    """The directory keeps only Crossref's title, e.g. "Algorithm 1016" without the subtitle that says what it is"""
    if not doi:
        return title
    try:
        with urllib.request.urlopen(f'https://api.crossref.org/works/{doi}', timeout=30) as response:
            subtitle = (json.load(response)['message'].get('subtitle') or [''])[0]
    except (OSError, ValueError, KeyError):
        return title
    return f'{title}: {subtitle}' if subtitle and subtitle.lower() not in title.lower() else title


def fetch_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'Accept': 'application/json'}), timeout=30) as r:
        return json.load(r)


def datacite_metadata(doi):
    a = fetch_json(f'https://api.datacite.org/dois/{doi}')['data']['attributes']
    publisher = a.get('publisher') or ''
    return {
        'doi': doi,
        'title': a['titles'][0]['title'],
        'authors': [c['name'] for c in a['creators']],
        'venue': (
            'arXiv'
            if doi.lower().startswith('10.48550/')
            else publisher.get('name', publisher.get('')) if isinstance(publisher, dict) else publisher
        ),
        'year': int(a['publicationYear']),
        'type': 'phdthesis' if a['types'].get('resourceTypeGeneral') == 'Dissertation' else 'misc',
    }


def paper_metadata(doi):
    """Title, authors ("Family, Given"), venue, year and BibTeX type of a DOI, from Crossref or else DataCite"""
    if doi.lower().startswith('10.48550/'):  # arXiv
        return datacite_metadata(doi)
    try:
        m = fetch_json(f'https://api.crossref.org/works/{doi}')['message']
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        return datacite_metadata(doi)
    title = m['title'][0] + (f": {m['subtitle'][0]}" if m.get('subtitle') else '')
    kind = {'journal-article': 'article', 'proceedings-article': 'inproceedings', 'book-chapter': 'incollection'}
    return {
        'doi': doi,
        'title': html.unescape(title),
        'authors': [f"{a['family']}, {a['given']}" if 'given' in a else a['family'] for a in m.get('author', [])],
        'venue': html.unescape((m.get('container-title') or [m.get('publisher', '')])[0]),
        'volume': m.get('volume'),
        'issue': m.get('issue'),
        'pages': m.get('page'),
        'year': (m.get('published-print') or m['issued'])['date-parts'][0][0],
        'type': kind.get(m['type'], 'misc'),
    }


def project_papers():
    """The metadata of every DOI in the gallery's `papers` lists; entries without a DOI are given in full there"""
    import yaml

    dois = sorted(
        {
            paper
            for section in yaml.safe_load(GALLERY.read_text(encoding='utf-8'))
            for project in section['projects']
            for paper in project.get('papers', [])
            if isinstance(paper, str)
        }
    )
    try:
        papers = {doi: {**paper_metadata(doi), **PAPER_CORRECTIONS.get(doi, {})} for doi in dois}
    except (OSError, ValueError, KeyError) as error:
        print(f'Could not fetch the project papers, keeping {PAPERS_OUT.name}: {error}', file=sys.stderr)
        return
    PAPERS_OUT.write_text(json.dumps(papers, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {len(papers)} project papers to {PAPERS_OUT}')


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
            'title': complete_title(html.unescape(p['title']), p['doi']),
            'authors': html.unescape(p['authors'] or ''),
            'year': p['publication_year'],
            'venue': html.unescape(p['journal'] or p['publisher'] or ''),
            'doi': p['doi'],
            'url': p['url'],
        }
        for p in publications.values()
    ]
    corrected = []
    for entry in entries:
        key = entry['doi'] or entry['url']
        if key in CORRECTIONS:
            if CORRECTIONS[key] is None:
                continue
            entry = {**entry, **CORRECTIONS[key]}
        corrected.append(entry)
    entries = corrected
    entries.sort(key=lambda entry: (-(entry['year'] or 0), entry['title'].lower()))
    OUT.write_text(json.dumps(entries, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {len(entries)} publications to {OUT}')


if __name__ == '__main__':
    main()
    project_papers()
