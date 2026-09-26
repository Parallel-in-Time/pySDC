# Sphinx configuration for the pySDC website, https://parallel-in-time.org/pySDC

import os
import sys
from pathlib import Path

ROOT = os.path.abspath('../../')
sys.path.insert(0, ROOT)
# The tutorials are executed in Jupyter kernels, which inherit this environment
os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, [ROOT, os.environ.get('PYTHONPATH')]))

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
    'myst_nb',
    'sphinx_design',
    'sphinx_copybutton',
    'sphinxemoji.sphinxemoji',
]

master_doc = 'index'
# The doc_*.rst files and the READMEs of the ported tutorial steps are only ever pulled into other pages with
# `.. include::`.
exclude_patterns = ['conf.py', '**/__init__.py', '**/doc_*.rst', 'tutorial/step_*/README.rst']

# Ported tutorials are jupytext "percent" scripts, linked into docs/source/tutorial. Sphinx runs them as notebooks.
nb_custom_formats = {'.py': ['jupytext.reads', {'fmt': 'py:percent'}]}
nb_execution_mode = 'force'
nb_execution_raise_on_error = True
nb_execution_timeout = 600
myst_enable_extensions = ['colon_fence', 'dollarmath', 'amsmath']
templates_path = ['_templates']
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
    'article_header_end': ['notebook-links'],
    'secondary_sidebar_items': ['page-toc', 'edit-this-page', 'sourcelink'],
    'footer_start': ['copyright'],
    'footer_end': ['theme-version'],
}
html_context = {
    'github_user': 'Parallel-in-Time',
    'github_repo': 'pySDC',
    'github_version': 'master',
    'doc_path': 'docs/source',
    # docs/source/tutorial/step_N links to pySDC/tutorial/step_N, where the ported tutorials really live
    'edit_page_url_template': (
        'https://github.com/Parallel-in-Time/pySDC/edit/master/'
        "{{ 'pySDC' if file_name.endswith('.py') else doc_path }}/{{ file_name }}"
    ),
}

INSTALL_CELL = """# Installs pySDC where it is missing, e.g. on Colab
try:
    import pySDC
except ImportError:
    %pip install -q git+https://github.com/Parallel-in-Time/pySDC.git"""


def write_notebooks(app, exception):
    """Next to each tutorial page, write the notebook that notebook-links.html offers for download and Colab"""
    if exception or app.builder.name != 'html':
        return
    import jupytext
    import nbformat

    for docname in app.env.found_docs:
        source = Path(app.env.doc2path(docname))
        if source.suffix == '.py':
            notebook = jupytext.read(source)
            notebook.cells.insert(0, nbformat.v4.new_code_cell(INSTALL_CELL))
            nbformat.write(notebook, Path(app.outdir) / f'{docname}.ipynb')


def setup(app):
    app.connect('build-finished', write_notebooks)
