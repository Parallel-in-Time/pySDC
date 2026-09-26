# Sphinx configuration for the pySDC website, https://parallel-in-time.org/pySDC

import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

project = 'pySDC'
copyright = '2014-2026, Robert Speck'
author = (
    'Robert Speck, Thibaut Lunet, Thomas Baumann, Lisa Wimmer, Ikrom Akramov, Giacomo Rosilho De Souza, '
    'Jakob Fritz, Jemma Shipton, Abdelouahed Ouardghi, Gayatri Čaklović'
)
version = '5.8'
release = '5.8'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_design',
    'sphinx_copybutton',
    'sphinxemoji.sphinxemoji',
]

master_doc = 'index'
# The doc_*.rst files are only ever pulled into the tutorial and project pages with `.. include::`.
exclude_patterns = ['**/doc_*.rst']
add_module_names = False
toc_object_entries_show_parents = 'hide'
suppress_warnings = ['image.nonlocal_uri']
autodoc_mock_imports = ['dolfin', 'mpi4py', 'petsc4py', 'mpi4py_fft', 'cupy', 'firedrake', 'gusto', 'vtk', 'vtkmodules']

html_theme = 'pydata_sphinx_theme'
html_title = 'pySDC'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    'logo': {'text': 'pySDC'},
    'icon_links': [
        {'name': 'GitHub', 'url': 'https://github.com/Parallel-in-Time/pySDC', 'icon': 'fa-brands fa-github'},
        {'name': 'PyPI', 'url': 'https://pypi.org/project/pySDC', 'icon': 'fa-brands fa-python'},
    ],
    'navbar_align': 'left',
    'show_nav_level': 1,
    'navigation_depth': 3,
    'show_toc_level': 2,
    'use_edit_page_button': True,
    'secondary_sidebar_items': ['page-toc', 'edit-this-page', 'sourcelink'],
    'footer_start': ['copyright'],
    'footer_end': ['theme-version'],
}
html_context = {
    'github_user': 'Parallel-in-Time',
    'github_repo': 'pySDC',
    'github_version': 'master',
    'doc_path': 'docs/source',
}
