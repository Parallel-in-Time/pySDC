# Sphinx configuration for the pySDC website, https://parallel-in-time.org/pySDC

import os
import re
import sys
from fnmatch import fnmatch
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
# `.. include::`, and the helper modules next to the tutorial parts are not notebooks.
exclude_patterns = [
    'conf.py',
    '**/__init__.py',
    '**/doc_*.rst',
    'tutorial/step_*/README.rst',
    'tutorial/step_*/HookClass_*.py',
    'tutorial/step_4/PenningTrap_3D_coarse.py',
    'tutorial/step_*/[a-z]*.py',  # helper modules, such as step_9/paradiag_setup.py
    'tutorial/step_7/F_2_*.py',  # the plotting script of part F
]

# Ported tutorials are jupytext "percent" scripts, linked into docs/source/tutorial. Sphinx runs them as notebooks.
nb_custom_formats = {'.py': ['jupytext.reads', {'fmt': 'py:percent'}]}
nb_execution_mode = 'force'
nb_execution_raise_on_error = True
nb_execution_timeout = 600
nb_execution_in_temp = True  # tutorials write data/ relative to where they run
myst_enable_extensions = ['colon_fence', 'dollarmath', 'amsmath']
templates_path = ['_templates']
add_module_names = False
toc_object_entries_show_parents = 'hide'
suppress_warnings = ['image.nonlocal_uri']
# Attributes sections become fields, so that they do not describe the attributes autodoc documents a second time
napoleon_use_ivar = True
# Every problem class has its own dtype_u and dtype_f, and "matrix" names no class: link the generic ones instead of
# letting Sphinx pick one of the ~70 candidates
napoleon_preprocess_types = True
napoleon_use_rtype = False  # the aliases do not reach the separate return-type field
napoleon_type_aliases = {
    'dtype_u': ':py:attr:`~pySDC.core.problem.Problem.dtype_u`',
    'dtype_f': ':py:attr:`~pySDC.core.problem.Problem.dtype_f`',
    'matrix': 'matrix',
}
autodoc_mock_imports = ['dolfin', 'mpi4py', 'petsc4py', 'mpi4py_fft', 'cupy', 'firedrake', 'gusto', 'vtk', 'vtkmodules']

html_theme = 'pydata_sphinx_theme'
html_title = 'pySDC'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = [('run-in-browser.js', {'type': 'module'})]

# Tutorials whose code runs in the browser, in Pyodide. Which ones can, and why the others cannot (MPI, FEniCS,
# PETSc, ...), was measured by running every tutorial there. The wheels are built by docs/update_apidocs.sh;
# without them, no page gets the button.
BROWSER_PAGES = ['tutorial/step_*/*']
# The parts that cannot, and why. They are not executed by the docs build either, as its environment lacks the same
# things; their pages show the results of the CI jobs that have them.
NOT_IN_BROWSER = {
    'tutorial/step_6/C_*': 'it runs on several processes with MPI (mpi4py).',
    'tutorial/step_7/A_*': 'it needs FEniCS.',
    'tutorial/step_7/B_*': 'it needs mpi4py-fft and MPI.',
    'tutorial/step_7/C_*': 'it needs PETSc (petsc4py) and MPI.',
    'tutorial/step_7/D_*': 'it needs PyTorch.',
    'tutorial/step_7/E_*': 'it needs Firedrake.',
    'tutorial/step_7/F_*': 'it needs Firedrake and Gusto.',
    'tutorial/step_7/G_*': 'it needs a GPU and CuPy.',
    'tutorial/step_9/E_*': 'it runs on several processes with MPI (mpi4py).',
}
nb_execution_excludepatterns = [f'{page}.py' for page in NOT_IN_BROWSER]
BROWSER_WHEELS = sorted(wheel.name for wheel in Path(__file__).parent.glob('_static/wheels/*.whl'))
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


def add_run_in_browser(app, pagename, templatename, context, doctree):
    reasons = [reason for page, reason in NOT_IN_BROWSER.items() if fnmatch(pagename, page)]
    context['run_unavailable'] = reasons[0] if reasons else None
    context['run_in_browser'] = (
        bool(BROWSER_WHEELS) and not reasons and any(fnmatch(pagename, page) for page in BROWSER_PAGES)
    )
    context['browser_wheels'] = BROWSER_WHEELS


def add_project_gallery(app, docname, source):
    """Replace the placeholder on the projects page with the cards and the toctree from pySDC/projects/gallery.yml"""
    if docname != 'projects/index':
        return
    import yaml

    gallery = Path(ROOT, 'pySDC/projects/gallery.yml')
    app.env.note_dependency(gallery)
    sections = yaml.safe_load(gallery.read_text(encoding='utf-8'))
    pages = [project['page'] for section in sections for project in section['projects']]
    missing = [page for page in pages if not Path(app.srcdir, 'projects', f'{page}.rst').exists()]
    if missing:
        raise ValueError(f'{gallery} names pages that docs/source/projects does not have: {missing}')
    lines = []
    for section in sections:
        lines += [section['section'], '-' * len(section['section']), '']
        lines += ['.. grid:: 1 2 2 3', '   :gutter: 3', '   :class-container: project-gallery', '']
        for project in section['projects']:
            lines += [
                f"   .. grid-item-card:: {project['title']}",
                f"      :link: {project['page']}",
                '      :link-type: doc',
            ]
            if 'image' in project:
                lines += [f"      :img-top: /../../{project['image']}", f"      :img-alt: {project['title']}"]
            else:
                lines += ['      :class-card: no-image']
            lines += ['', f"      {project['summary']}", '']
    lines += ['.. toctree::', '   :hidden:', '']
    lines += [f'   {page}' for page in pages]
    source[0] = source[0].replace('.. project-gallery', '\n'.join(lines))


# The API overview on api.rst: one table per role in a run, with the first paragraph of each class's docstring. The
# classes are read with ast, so that modules whose imports are missing here (FEniCS, PETSc, ...) are listed too.
API_CATEGORIES = [
    (
        'Problems',
        'implementations/problem_classes',
        ['Problem'],
        'The equations: right-hand sides, implicit solves and, where known, exact solutions.',
    ),
    (
        'Sweepers',
        'implementations/sweeper_classes',
        ['Sweeper'],
        'The integrators within a step: SDC with its preconditioners, IMEX and multi-implicit splittings, Runge-Kutta.',
    ),
    (
        'Controllers',
        'implementations/controller_classes',
        ['Controller'],
        'Run the steps, one after the other or in parallel, with SDC, MLSDC, PFASST or ParaDiag.',
    ),
    (
        'Convergence controllers',
        'implementations/convergence_controller_classes',
        ['ConvergenceController'],
        'Change a run while it goes: error estimates, adaptive step sizes, stopping criteria and restarts.',
    ),
    ('Hooks', 'implementations/hooks', ['Hooks'], 'Record what happens during a run into the statistics.'),
    (
        'Transfer',
        'implementations/transfer_classes',
        ['SpaceTransfer', 'BaseTransfer'],
        'Move data between the levels of MLSDC and PFASST, in space and between them.',
    ),
    ('Data types', 'implementations/datatype_classes', None, 'What solutions and right-hand sides are stored in.'),
    ('Core', 'core', None, 'The base classes everything above derives from, and the step and level they run on.'),
]


def _summary(docstring):
    """The first paragraph of a docstring, on one line and without footnote references"""
    paragraph = re.split(r'\n\s*\n', (docstring or '').strip())[0]
    if paragraph.startswith('..'):
        return ''
    text = ' '.join(paragraph.split())
    text = re.sub(r'\s*\[[#\w]+\]_', '', text)
    text = re.sub(r'\[([^\]]+)\]\((https?://(?:[^()\s]|\([^()\s]*\))+)\)', r'`\1 <\2>`__', text)
    # the first sentence: a period before a capital letter, not inside inline markup
    ends = [m.start() + 1 for m in re.finditer(r'\.\s+(?=[A-Z])', text) if text[: m.start()].count('`') % 2 == 0]
    text = text[: ends[0]] if ends else text
    return text[:-1] + '.' if text.endswith(':') else text


def _api_table(rows, header):
    lines = ['.. list-table::', '   :header-rows: 1', '   :widths: 35 65', '   :width: 100%', '   :class: api-overview']
    lines += ['', f'   * - {header}', '     - Summary']
    for name, summary in rows:
        lines += [f'   * - {name}', f'     - {summary or "—"}']
    return lines + ['']


def add_api_overview(app, docname, source):
    """Replace the placeholder on api.rst with the tables of API_CATEGORIES and of the helpers"""
    if docname != 'api':
        return
    import ast

    files = sorted(Path(ROOT, 'pySDC').glob('*/**/*.py'))
    files = [f for f in files if f.parts[-2] in ('core', 'helpers') or 'implementations' in f.parts]
    classes, bases = [], {}
    for file in files:
        app.env.note_dependency(file)
        tree = ast.parse(file.read_text(encoding='utf-8'))
        module = '.'.join(file.relative_to(ROOT).with_suffix('').parts)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases[node.name] = [getattr(base, 'id', getattr(base, 'attr', '')) for base in node.bases]
                if not node.name.startswith('_'):
                    classes.append((file, module, node.name, _summary(ast.get_docstring(node))))

    def derives(name, roots, seen=()):
        return name in roots or any(derives(b, roots, seen + (name,)) for b in bases.get(name, []) if b not in seen)

    lines = []
    for title, folder, roots, description in API_CATEGORIES:
        rows = [
            (f':py:class:`~{module}.{name}`', summary)
            for file, module, name, summary in classes
            if file.parent == Path(ROOT, 'pySDC', folder) and (roots is None or derives(name, roots))
        ]
        lines += [title, '-' * len(title), '', description, ''] + _api_table(rows, 'Class')
    rows = []
    for file in files:
        if file.parent == Path(ROOT, 'pySDC', 'helpers') and file.name != '__init__.py':
            tree = ast.parse(file.read_text(encoding='utf-8'))
            names = [
                node.name
                for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith('_')
            ]
            module = '.'.join(file.relative_to(ROOT).with_suffix('').parts)
            rows.append((f':py:mod:`~{module}`', ', '.join(f'``{name}``' for name in names)))
    lines += ['Helpers', '-------', '', 'Utilities for statistics, plots, setups, input and output.', '']
    lines += _api_table(rows, 'Module')
    source[0] = source[0].replace('.. api-overview', '\n'.join(lines))


def setup(app):
    app.connect('build-finished', write_notebooks)
    app.connect('html-page-context', add_run_in_browser)
    app.connect('source-read', add_project_gallery)
    app.connect('source-read', add_api_overview)
