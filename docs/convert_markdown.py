#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 19:47:56 2023

@author: telu
"""

import os
import re
import glob
import json
import m2r2
import numpy as np

mdFiles = ['README.md', 'CONTRIBUTING.md', 'CHANGELOG.md', 'CODE_OF_CONDUCT.md', 'docs/contrib']

docSources = 'docs/source'


counter = np.array(0)

with open('docs/emojis.json') as f:
    emojis = set(json.load(f).keys())


def wrappEmojis(rst):
    for emoji in emojis:
        rst = rst.replace(emoji, f'|{emoji}|')
    return rst


def addSectionRefs(rst, baseName):
    sections = {}
    lines = rst.splitlines()
    # Search for sections in rst file
    for i in range(len(lines) - 2):
        conds = [
            len(lines[i + 1]) and lines[i + 1][0] in ['=', '-', '^', '"'],
            lines[i + 2] == lines[i - 1] == '',
            len(lines[i]) == len(lines[i + 1]),
        ]
        if all(conds):
            sections[i] = lines[i]
    # Add unique references before each section
    for i, title in sections.items():
        ref = '-'.join([elt for elt in title.lower().split(' ') if elt != ''])
        for char in ['#', "'", '^', '°', '!']:
            ref = ref.replace(char, '')
        ref = f'{baseName}/{ref}'
        lines[i] = f'.. _{ref}:\n\n' + lines[i]
    # Returns all concatenated lines
    return '\n'.join(lines)


def completeRefLinks(rst, baseName):
    i = 0
    while i != -1:
        i = rst.find(':ref:`', i)
        if i != -1:
            iLink = rst.find('<', i)
            rst = rst[: iLink + 1] + f'{baseName}/' + rst[iLink + 1 :]
            i += 6
    return rst


def linkLocalAnchors(rst, baseName):
    """addSectionRefs replaces a section's id by its label, so point the page's own #anchor links to the label"""
    labels = set(re.findall(rf'^\.\. _{re.escape(baseName)}/(\S+):$', rst, re.M))

    def toRef(match):
        text, anchor = match.groups()
        return f':ref:`{text} <{anchor}>`' if anchor in labels else match.group(0)  # completeRefLinks adds baseName

    return re.sub(r'`([^`<]+?) <#([^>]+)>`_', toRef, rst)


def addOrphanTag(rst):
    return '\n:orphan:\n\n' + rst


def setImgPath(rst, md):
    """Raw <img> tags of docs/img, which conf.py's html_static_path copies into _static"""
    return rst.replace('<img src="./docs/img/', f'<img src="{"../" * md.count("/")}_static/')


def linkReadmeToIndex(rst):
    return rst.replace('<./README>', '<./index>')


def titleOverview(rst):
    """On the website, the README is the Overview, as the navigation bar calls it; GitHub keeps its welcome"""
    return rst.replace('Welcome to pySDC!\n=================\n', 'Overview\n========\n', 1)


def linkFilesToGitHub(text, md):
    """m2r2 turns every relative link into a :doc: reference, which only works for other Markdown pages"""

    def toGitHub(match):
        path = os.path.normpath(os.path.join(os.path.dirname(md), match.group(2)))
        return f'{match.group(1)}(https://github.com/Parallel-in-Time/pySDC/blob/master/{path})'

    return re.sub(r'((?<!!)\[[^\]]*\])\((?!<|\w+://|#|mailto:)([^)\s#]+(?<!\.md))\)', toGitHub, text)


def dollarMath(text):
    """m2r2 leaves GitHub's $...$ and $$...$$ math as text, but passes on math code blocks and :math: roles"""
    parts = re.split(r'(^```.*?^```$|`[^`\n]*`)', text, flags=re.M | re.S)  # leave code alone
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'^\$\$\n(.*?)\n\$\$$', r'```math\n\1\n```', parts[i], flags=re.M | re.S)
        parts[i] = re.sub(r'(?<![\w$\\])\$(?=\S)([^$\n]+?)(?<=\S)\$(?![\w$])', r':math:`\1`', parts[i])
    return ''.join(parts)


def convert(md, orphan=False, sectionRefs=True):
    baseName = os.path.splitext(md)[0]
    with open(md) as f:
        rst = m2r2.convert(dollarMath(linkFilesToGitHub(f.read(), md)), parse_relative_links=True)
    rst = wrappEmojis(rst)
    if sectionRefs:
        rst = addSectionRefs(rst, baseName)
        rst = linkLocalAnchors(rst, baseName)
    rst = completeRefLinks(rst, baseName)
    if orphan:
        rst = addOrphanTag(rst)
    rst = setImgPath(rst, md)
    rst = linkReadmeToIndex(rst)
    if md == 'README.md':
        rst = titleOverview(rst)
    with open(f'{docSources}/{baseName}.rst', 'w') as f:
        f.write(rst)
    print(f'Converted {md} to {docSources}/{baseName}.rst')


for md in mdFiles:
    if os.path.isfile(md):
        isNotMain = md != 'README.md'
        convert(md, orphan=isNotMain, sectionRefs=isNotMain)
    elif os.path.isdir(md):
        os.makedirs(f'{docSources}/{md}', exist_ok=True)
        for f in glob.glob(f'{md}/*.md'):
            convert(f, orphan=True)
    else:
        raise ValueError('{md} is not a md file or a folder')
