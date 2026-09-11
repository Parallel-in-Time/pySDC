#!/usr/bin/env python3
"""Estimate the AI-authorship split of the pySDC code, to keep aidecl.yaml honest.

Every surviving line of tracked code is attributed to the commit that last touched it
(``git blame -w``), that commit is classified, and the three ``code_proportion``
percentages are reported and compared against the ones declared in ``aidecl.yaml``.

What the classification can and cannot see: it reads commit trailers and pull request
metadata, so it is only as good as those. A commit written with AI help but pushed
without a ``Co-authored-by`` trailer looks human, which is what OVERRIDES is for.
Treat the output as a ballpark to inform an edit of aidecl.yaml, not as ground truth.

Needs ``gh`` (authenticated) for the pull request reviews behind the ai_assisted
bucket. Without it, that bucket is trailer-only and comes out too low.

Usage:
    python etc/aidecl_audit.py                     # audit the whole code base
    python etc/aidecl_audit.py --since 2026-01-01  # only recently touched code
    python etc/aidecl_audit.py --selftest          # check the classifier itself
"""

import argparse
import collections
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

CODE_GLOBS = ['*.py', '*.pyx', '*.sh', '*.yml', '*.yaml', '*.toml', '*.cfg', '*.ini']

CLAUDE = re.compile(r'claude', re.I)
COPILOT = re.compile(r'copilot', re.I)
PR_NUMBER = re.compile(r'\(#(\d+)\)\s*$')
# git blame --porcelain: a line group starts with "<sha> <orig> <final> <numlines>"
BLAME_GROUP = re.compile(rb'^([0-9a-f]{40}) \d+ \d+ (\d+)$')

GENERATED, ASSISTED, HUMAN = 'ai_generated', 'ai_assisted', 'human_only'
UNCOMMITTED = '0' * 40  # what git blame reports for lines not committed yet

# Commits whose AI involvement is not recorded anywhere in git. Keep this list short:
# every entry is a reminder that a trailer was missing when the work was pushed.
OVERRIDES = {
    # "PINN and FNO playgrounds", pushed straight to master without a trailer.
    '20957fd5b9b3c3f7148fbf8e56c23e958ae1d8eb': GENERATED,
}


def run(*args, binary=False):
    out = subprocess.run(args, capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode()


def classify(commit, copilot_authored, copilot_reviewed):
    """Bucket one commit. `commit` is a dict with sha, author, subject and body."""
    if commit['sha'] in OVERRIDES:
        return OVERRIDES[commit['sha']]
    match = PR_NUMBER.search(commit['subject'])
    pr = int(match.group(1)) if match else None
    if CLAUDE.search(commit['body']) or pr in copilot_authored or COPILOT.match(commit['author']):
        return GENERATED
    if pr in copilot_reviewed or COPILOT.search(commit['body']):
        return ASSISTED
    return HUMAN


def blame_all():
    """Map commit sha -> number of lines in the current tree last touched by it."""
    files = [f for f in run('git', 'ls-files', '-z', *CODE_GLOBS).split('\0') if f]

    def blame(path):
        counts = collections.Counter()
        try:
            output = run('git', 'blame', '-w', '--porcelain', '--', path, binary=True)
        except subprocess.CalledProcessError:
            return counts  # submodule, symlink, whatever: not our code
        for line in output.splitlines():
            group = BLAME_GROUP.match(line)
            if group:
                counts[group.group(1).decode()] += int(group.group(2))
        return counts

    total = collections.Counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for counts in pool.map(blame, files):
            total.update(counts)
    uncommitted = total.pop(UNCOMMITTED, 0)
    if uncommitted:
        print(f'warning: ignoring {uncommitted} uncommitted lines, commit them for an exact count.\n', file=sys.stderr)
    return total


def commit_metadata(shas):
    """Fetch author, date, subject and body for many commits, a few calls not many."""
    meta = {}
    for i in range(0, len(shas), 200):
        blob = run('git', 'show', '--no-patch', '--format=@@@%H%n%ai%n%an%n%s%n%b%n@@END@@', *shas[i : i + 200])
        for block in blob.split('@@@')[1:]:
            lines = block.split('\n')
            meta[lines[0].strip()] = {
                'sha': lines[0].strip(),
                'date': lines[1][:10],
                'author': lines[2],
                'subject': lines[3],
                'body': block,
            }
    return meta


def copilot_pull_requests():
    """(authored by the Copilot agent, reviewed by Copilot) pull request numbers."""
    query = 'gh pr list --state merged --base master --limit 1000 --json number,author,reviews'
    try:
        raw = run(*query.split())
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('warning: `gh` unavailable, so ai_assisted counts trailers only and reads low.\n', file=sys.stderr)
        return set(), set()
    prs = json.loads(raw)
    authored = {p['number'] for p in prs if COPILOT.search((p['author'] or {}).get('login', ''))}
    reviewed = {
        p['number'] for p in prs if any(COPILOT.search((r['author'] or {}).get('login', '')) for r in p['reviews'])
    }
    return authored, reviewed


def declared_percentages():
    """The three percentages currently in aidecl.yaml, so we can report drift."""
    try:
        text = open('aidecl.yaml').read()
    except OSError:
        return {}
    # A regex rather than a yaml import, to keep this script dependency-free.
    return {key: float(value) for key, value in re.findall(r'^\s*(\w+)_percent:\s*([\d.]+)\s*$', text, re.M)}


def audit(since=None):
    lines_per_commit = blame_all()
    meta = commit_metadata(list(lines_per_commit))
    authored, reviewed = copilot_pull_requests()

    totals = collections.Counter()
    for sha, lines in lines_per_commit.items():
        commit = meta.get(sha)
        if commit is None:
            totals[HUMAN] += lines
            continue
        if since and commit['date'] < since:
            continue
        totals[classify(commit, authored, reviewed)] += lines

    grand_total = sum(totals.values())
    if not grand_total:
        print('no code matched, nothing to report')
        return 0

    scope = f'code last touched since {since}' if since else 'the whole code base'
    print(f'{grand_total} lines of tracked code ({scope}):\n')
    declared = declared_percentages()
    for bucket in (GENERATED, ASSISTED, HUMAN):
        measured = 100 * totals[bucket] / grand_total
        drift = ''
        if bucket in declared and not since:
            gap = measured - declared[bucket]
            drift = f'   declared {declared[bucket]:g}%  ({gap:+.1f})'
        print(f'  {bucket:14} {totals[bucket]:>7}  {measured:5.1f}%{drift}')

    if since:
        print('\nPercentages are for this slice only; aidecl.yaml declares the whole code base.')
    return 0


def selftest():
    """One runnable check on the classification, the only non-obvious logic here."""
    authored, reviewed = {600}, {618}

    def commit(sha='0' * 40, author='Robert Speck', subject='Fix a thing', body=''):
        return {'sha': sha, 'author': author, 'subject': subject, 'body': body}

    cases = [
        (commit(body='Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>'), GENERATED),
        (commit(subject='Fix CI (#600)'), GENERATED),  # Copilot agent authored the PR
        (commit(author='Copilot', subject='Add a thing'), GENERATED),
        (commit(subject='Add heat equation (#618)'), ASSISTED),  # Copilot only reviewed it
        (commit(body='* Implemented suggestions from copilot'), ASSISTED),
        (commit(subject='Add a sweeper (#599)'), HUMAN),
        (commit(), HUMAN),
        (commit(sha=next(iter(OVERRIDES))), OVERRIDES[next(iter(OVERRIDES))]),
    ]
    for case, expected in cases:
        actual = classify(case, authored, reviewed)
        assert actual == expected, f'{case["subject"]!r} -> {actual}, expected {expected}'

    # A Claude trailer wins over a pull request that Copilot merely reviewed.
    assert classify(commit(subject='Thing (#618)', body='Co-Authored-By: Claude'), authored, reviewed) == GENERATED
    print(f'selftest: {len(cases) + 1} cases passed')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--since', metavar='YYYY-MM-DD', help='only count code last touched on or after this date')
    parser.add_argument('--selftest', action='store_true', help='check the classifier and exit')
    args = parser.parse_args()
    sys.exit(selftest() if args.selftest else audit(args.since))
