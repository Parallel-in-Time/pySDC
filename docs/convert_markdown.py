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

mdFiles = [
    'README.md',
    'CONTRIBUTING.md',
    'CHANGELOG.md',
    'CODE_OF_CONDUCT.md',
    'docs/contrib']

docSources = 'docs/source'

with open('docs/emojis.json') as f:
    emojis = set(json.load(f).keys())

def convert(md):
    rst = m2r2.parse_from_file(md, parse_relative_links=True)
    for emoji in emojis:
        rst = rst.replace(emoji, f'|{emoji}|')
    baseName = os.path.splitext(md)[0]
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
