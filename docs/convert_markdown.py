#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 19:47:56 2023

@author: telu
"""
import os
import glob
import json
import m2r2
import numpy as np

mdFiles = [
    'README.md',
    'CONTRIBUTING.md',
    'CHANGELOG.md',
    'CODE_OF_CONDUCT.md',
    'docs/contrib']

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
    for i in range(len(lines)-2):
        conds = [
            len(lines[i+1]) and lines[i+1][0] in ['=', '-', '^', '"'],
            lines[i+2] == lines[i-1] == '',
            len(lines[i]) == len(lines[i+1])]
        if all(conds):
            sections[i] = lines[i]
    # Add unique references before each section
    for i, title in sections.items():
        ref = '-'.join([elt for elt in title.lower().split(' ') if elt != ''])
        for char in ['#', "'", '^', '°', '!']:
            ref = ref.replace(char, '')
        ref = f'{baseName}/{ref}'
        lines[i] = f'.. _{ref}:\n\n'+lines[i]
    # Returns all concatenated lines
    return '\n'.join(lines)

def completeRefLinks(rst, baseName):
    i = 0
    while i != -1:
        i = rst.find(':ref:`', i)
        if i != -1:
            iLink = rst.find('<', i)
            rst = rst[:iLink+1]+f'{baseName}/'+rst[iLink+1:]
            i += 6
    return rst

def convert(md):
    baseName = os.path.splitext(md)[0]
    rst = m2r2.parse_from_file(md, parse_relative_links=True)
    rst = wrappEmojis(rst)
    rst = addSectionRefs(rst, baseName)
    rst = completeRefLinks(rst, baseName)
    with open(f'{docSources}/{baseName}.rst', 'w') as f:
        f.write(rst)
    print(f'Converted {md} to {docSources}/{baseName}.rst')

for md in mdFiles:
    if os.path.isfile(md):
        convert(md)
    elif os.path.isdir(md):
        os.makedirs(f'{docSources}/{md}', exist_ok=True)
        for f in glob.glob(f'{md}/*.md'):
            convert(f)
    else:
        raise ValueError('{md} is not a md file or a folder')
