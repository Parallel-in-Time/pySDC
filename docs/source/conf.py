# Sphinx configuration for the pySDC website, https://parallel-in-time.org/pySDC

import hashlib
import inspect
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
version = '5.9'
release = '5.9'

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
# the source of members imported from mocked modules (Firedrake, Gusto, ...) is not there to show
viewcode_follow_imported_members = False

html_theme = 'pydata_sphinx_theme'
html_title = 'pySDC'
# the favicons: _templates/layout.html
html_static_path = ['_static', '../img']  # ../img: the logos in README.md
pygments_dark_style = 'github-dark'  # the theme's default dark style is loud
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
# of those, the parts Colab can run as they are (PyTorch is preinstalled there): they keep their Colab button
COLAB_TOO = ['tutorial/step_7/D_*']
BROWSER_WHEELS = sorted(wheel.name for wheel in Path(__file__).parent.glob('_static/wheels/*.whl'))
html_theme_options = {
    'logo': {'image_light': '_static/pysdc-logo.svg', 'image_dark': '_static/pysdc-logo-dark.svg', 'alt_text': 'pySDC'},
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
    'default_mode': 'auto',  # the theme's default is empty, which it reports as an error in every page's console
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
    context['colab'] = not reasons or any(fnmatch(pagename, page) for page in COLAB_TOO)
    context['run_in_browser'] = (
        bool(BROWSER_WHEELS) and not reasons and any(fnmatch(pagename, page) for page in BROWSER_PAGES)
    )
    context['browser_wheels'] = BROWSER_WHEELS


def add_section_title(app, pagename, templatename, context, doctree):
    """The sidebar's title: the top-level section this page is in, e.g. Tutorial, from the toctree"""
    relations = app.env.collect_relations()
    page = pagename
    while relations.get(page, [None])[0] not in (None, app.config.root_doc):
        page = relations[page][0]
    if relations.get(page, [None])[0] == app.config.root_doc:
        context['section_page'] = page
        context['section_title'] = app.env.titles[page].astext()


def _excerpt(app, excerpt):
    """The literalinclude of a gallery card's excerpt; the texts that delimit a passage of code have to be in the file"""
    lines = [
        f"      .. literalinclude:: /../../{excerpt['file']}",
        f"         :language: {excerpt.get('language', 'python')}",
    ]
    if 'lines' in excerpt:  # an output the tests write in CI, which the docs job fails on if it is missing
        return lines + [f"         :lines: {excerpt['lines']}"]
    file = Path(ROOT, excerpt['file'])
    app.env.note_dependency(file)
    text = file.read_text(encoding='utf-8')
    for option in ('start-at', 'end-at', 'end-before'):
        if option in excerpt:
            if excerpt[option] not in text:
                raise ValueError(f"The gallery excerpt of {excerpt['file']} needs {excerpt[option]!r}, which is gone")
            lines.append(f'         :{option}: {excerpt[option]}')
    return lines + ['         :dedent:']


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
            elif 'excerpt' in project:
                lines += ['      :class-card: excerpt-card', ''] + _excerpt(app, project['excerpt'])
            else:
                lines += ['      :class-card: no-image']
            lines += ['', f"      {project['summary']}", '']
    lines += ['.. toctree::', '   :hidden:', '']
    lines += [f'   {page}' for page in pages]
    source[0] = source[0].replace('.. project-gallery', '\n'.join(lines))


# The API overview on api.rst: one table per role in a run, with the first sentence of each class's docstring. The
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
        'The integrators within a step: SDC with its preconditioners and splittings, second-order and multistep methods, '
        'Runge-Kutta and ParaDiag.',
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
        'Move data between the levels of MLSDC and PFASST: restriction and interpolation in space, and the FAS correction.',
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


def _bibtex(kind, key, fields):
    body = ',\n'.join(f'    {name} = {{{value}}}' for name, value in fields.items() if value)
    return [f'      @{kind}{{{key},'] + ['      ' + line for line in body.split('\n')] + ['      }']


def add_project_papers(app, docname, source):
    """Append the papers done with a project, from `papers` in pySDC/projects/gallery.yml, to the project's page"""
    if not docname.startswith('projects/') or docname == 'projects/index':
        return
    import json
    import yaml

    gallery, metadata_file = Path(ROOT, 'pySDC/projects/gallery.yml'), Path(app.srcdir, 'project_papers.json')
    app.env.note_dependency(gallery)
    app.env.note_dependency(metadata_file)
    page = docname.split('/', 1)[1]
    projects = [
        project for section in yaml.safe_load(gallery.read_text(encoding='utf-8')) for project in section['projects']
    ]
    papers = next((project.get('papers', []) for project in projects if project['page'] == page), [])
    if not papers:
        return
    metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
    missing = [paper for paper in papers if isinstance(paper, str) and paper not in metadata]
    if missing:
        raise ValueError(f'{metadata_file.name} has no metadata for {missing}: run docs/update_publications.py')

    lines = ['', '', 'Papers', '------', '', 'The results of this project are published in:', '']
    for paper in papers:
        paper = metadata[paper] if isinstance(paper, str) else paper
        names = ', '.join(' '.join(reversed(author.split(', '))) for author in paper['authors'])
        venue = ('PhD thesis, ' if paper['type'] == 'phdthesis' else '') + f"*{paper['venue']}*"
        venue += f" {paper['volume']}" if paper.get('volume') else ''
        venue += f"({paper['issue']})" if paper.get('issue') else ''
        venue += f", {paper['pages'].replace('-', '–')}" if paper.get('pages') else ''
        link = f"https://doi.org/{paper['doi']}" if paper.get('doi') else paper['url']
        key = re.sub(r'\W', '', paper['authors'][0].split(',')[0].lower()) + str(paper['year'])
        key += re.sub(r'\W', '', paper['title'].split()[0].lower())
        fields = {
            'author': ' and '.join(paper['authors']),
            'title': paper['title'],
            {'article': 'journal', 'phdthesis': 'school', 'mastersthesis': 'school', 'misc': 'howpublished'}.get(
                paper['type'], 'booktitle'
            ): paper['venue'],
            'volume': paper.get('volume'),
            'number': paper.get('issue'),
            'pages': (paper.get('pages') or '').replace('-', '--'),
            'year': paper['year'],
            'doi': paper.get('doi'),
            'url': None if paper.get('doi') else paper['url'],
        }
        lines += [f"- {names}, **{paper['title']}**, {venue}, {paper['year']}, {link}", '']
        lines += ['  .. dropdown:: BibTeX', '     :class-container: paper-bibtex', '']
        lines += ['     .. code-block:: bibtex', '']
        lines += ['   ' + line for line in _bibtex(paper['type'], key, fields)] + ['']
    source[0] += '\n'.join(lines)


def add_publications(app, docname, source):
    """Replace the placeholder on publications.rst with how to cite pySDC and the publications that use it"""
    if docname != 'publications':
        return
    import json
    import yaml

    cff_file, publications_file = Path(ROOT, 'CITATION.cff'), Path(app.srcdir, 'publications.json')
    app.env.note_dependency(cff_file)
    app.env.note_dependency(publications_file)
    cff = yaml.safe_load(cff_file.read_text(encoding='utf-8'))
    paper = cff['preferred-citation']

    def names(authors, bibtex=False):
        return (' and ' if bibtex else ', ').join(
            f"{a['family-names']}, {a['given-names']}" if bibtex else f"{a['given-names']} {a['family-names']}"
            for a in authors
        )

    lines = [
        'Cite pySDC',
        '----------',
        '',
        "If you use pySDC for your work, please cite the paper, and the version of the software you used.",
        '',
    ]
    lines += ['.. grid:: 1 1 1 1', '   :gutter: 3', '']
    lines += ['   .. grid-item-card:: The paper', '']
    lines += [
        f"      {names(paper['authors'])}, **{paper['title']}**, *{paper['journal']}* {paper['volume']}({paper['issue']}),"
    ]
    lines += [f"      {paper['start']}–{paper['end']}, {paper['year']}, https://doi.org/{paper['doi']}", '']
    lines += ['      .. code-block:: bibtex', '']
    lines += [
        '   ' + line
        for line in _bibtex(
            'article',
            'speck2019pysdc',
            {
                'author': names(paper['authors'], bibtex=True),
                'title': paper['title'].replace('pySDC', '{pySDC}').replace('—', '---'),
                'journal': paper['journal'],
                'volume': paper['volume'],
                'number': paper['issue'],
                'pages': f"{paper['start']}--{paper['end']}",
                'year': paper['year'],
                'doi': paper['doi'],
            },
        )
    ]
    lines += ['', f"   .. grid-item-card:: The software, version {cff['version']}", '']
    lines += [
        f"      {names(cff['authors'])}, **{cff['title']}**, version {cff['version']}, {cff['date-released'].year},"
    ]
    lines += [f"      https://doi.org/{cff['doi']}", '']
    lines += [
        '      This DOI always leads to the latest version. For the DOI of the exact version you used, see the list'
    ]
    lines += [f"      of versions on `Zenodo <https://doi.org/{cff['doi']}>`__.", '']
    lines += ['      .. code-block:: bibtex', '']
    lines += [
        '   ' + line
        for line in _bibtex(
            'software',
            'pysdc',
            {
                'author': names(cff['authors'], bibtex=True),
                'title': cff['title'],
                'version': cff['version'],
                'year': cff['date-released'].year,
                'doi': cff['doi'],
                'url': cff['repository-code'],
            },
        )
    ]

    publications = json.loads(publications_file.read_text(encoding='utf-8'))
    lines += ['', 'Publications using pySDC', '------------------------', '']
    lines += ['Research that mentions or cites pySDC, as listed in the `Helmholtz Research Software Directory']
    lines += ['<https://helmholtz.software/software/pysdc>`__. To add a publication, add it as a mention there.', '']
    year = None
    for publication in publications:
        if publication['year'] != year:
            year = publication['year']
            lines += [f'.. rubric:: {year or "Undated"}', '']
        link = f"https://doi.org/{publication['doi']}" if publication['doi'] else publication['url']
        title = f"`{publication['title']} <{link}>`__" if link else publication['title']
        venue = f", *{publication['venue']}*" if publication['venue'] else ''
        lines += [f"- {publication['authors'] or ''}: {title}{venue}", '']
    source[0] = source[0].replace('.. publications-page', '\n'.join(lines))


LANDING_DEMO = """
.. raw:: html

   <div class="landing-demo" data-wheels="{wheels}" data-wheels-url="_static/wheels/">
     <form>
       <label><span>Time step Δt</span>
         <select name="dt"><option>0.001</option><option>0.01</option><option selected>0.1</option><option>1</option>
         </select></label>
       <label><span>Preconditioner Q<sub>Δ</sub></span>
         <select name="QI">
           <option value="IE">implicit Euler (IE)</option>
           <option value="LU" selected>LU trick (LU)</option>
           <option value="MIN-SR-S">MIN-SR-S (diagonal)</option>
           <option value="MIN-SR-NS">MIN-SR-NS (diagonal)</option>
           <option value="MIN-SR-FLEX">MIN-SR-FLEX (diagonal, per sweep)</option>
           <option value="PIC">Picard (explicit)</option>
         </select></label>
       <label><span>Collocation nodes M</span>
         <select name="num_nodes"><option>2</option><option selected>3</option><option>4</option><option>5</option>
         </select></label>
       <label><span>Levels</span>
         <select name="levels"><option value="1" selected>1 (SDC)</option><option value="2">2 (MLSDC)</option>
         </select></label>
       <label><span>Interface width ε</span>
         <select name="eps"><option>0.02</option><option selected>0.04</option><option>0.08</option></select></label>
       <div class="demo-buttons">
         <button type="submit" class="btn btn-sm demo-run"><i class="fa-solid fa-play"></i> Run</button>
         <button type="button" class="btn btn-sm demo-clear" disabled>Clear</button>
       </div>
     </form>
     <p class="demo-status">{status}</p>
     <img class="demo-plot" alt="Residual over the iterations of the runs so far" hidden>
   </div>
   <script type="module" src="_static/landing-demo.js?v={version}"></script>
"""


def add_landing_demo(app, docname, source):
    """Replace the placeholder on the landing page with the demo's form, which landing-demo.js runs"""
    if docname != 'index':
        return
    status = (
        'Choose a setup and press Run. Each run adds a curve, so you can compare setups.'
        if BROWSER_WHEELS
        else 'The demo needs the pySDC wheels, which docs/update_apidocs.sh builds.'
    )
    # the ?v= Sphinx gives its own scripts, so that browsers fetch the script again when it changes
    version = hashlib.md5(Path(app.srcdir, '_static', 'landing-demo.js').read_bytes()).hexdigest()[:8]
    demo = LANDING_DEMO.format(wheels=' '.join(BROWSER_WHEELS), status=status, version=version)
    if not BROWSER_WHEELS:
        demo = demo.replace('class="btn btn-sm demo-run"', 'class="btn btn-sm demo-run" disabled')
    source[0] = source[0].replace('.. landing-demo', demo)


def link_tutorial_parts(app, docname, source):
    """On a step's page, make each "Part X: ..." of its README the link to that part, instead of a second list"""
    match = re.fullmatch(r'tutorial/(step_\d+)', docname)
    if not match:
        return
    step = match.group(1)
    readme = Path(app.srcdir, 'tutorial', step, 'README.rst')
    app.env.note_dependency(readme)
    parts = {Path(entry).name[0]: entry for entry in re.findall(rf'^\s+({step}/[A-Z]_\S+)$', source[0], re.M)}

    def link(m):
        letter, title, rest = m.groups()
        if letter not in parts:
            return m.group(0)
        return f'- :doc:`Part {letter}: {title} <{parts[letter]}>`.{rest}'

    text = re.sub(r'^- \*\*Part ([A-Z]): (.+?)\.\*\*(.*)$', link, readme.read_text(encoding='utf-8'), flags=re.M)
    source[0] = (
        source[0].replace(f'.. include:: {step}/README.rst', text).replace('.. toctree::', '.. toctree::\n   :hidden:')
    )


def skip_modules(app, what, name, obj, skip, options):
    """Class attributes such as `xp = numpy` would be documented as the module, with the path it was imported from"""
    return True if inspect.ismodule(obj) else None


def setup(app):
    app.connect('build-finished', write_notebooks)
    app.connect('html-page-context', add_run_in_browser)
    app.connect('html-page-context', add_section_title)
    app.connect('source-read', add_project_gallery)
    app.connect('source-read', add_api_overview)
    app.connect('source-read', add_publications)
    app.connect('source-read', add_project_papers)
    app.connect('source-read', add_landing_demo)
    app.connect('autodoc-skip-member', skip_modules)
    app.connect('source-read', link_tutorial_parts)
    # the hooks only read the environment or rewrite the source they are given, so sphinx-build -j can execute
    # the tutorials in parallel
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
