# coding=utf-8
import os
import os.path
import nose.tools
import pep8

BASE_PATH = os.path.abspath(os.path.join(__file__, '..', '..', '..'))


def check_files(dir):
    style = pep8.StyleGuide()
    style.options.max_line_length = 120
    style.options.ignore = 'E402'
    python_files = []
    for root, _, files in os.walk(dir):
        if 'pySDC/playgrounds/deprecated' not in root:
            python_files += [os.path.join(root, f) for f in files if f.endswith('.py')]

    for file in python_files:
        report = style.check_files([os.path.join(BASE_PATH, file)])
        report.print_statistics()
        nose.tools.assert_equal(report.total_errors, 0, "File %s has some PEP8 errors: %d" % (file, report.total_errors))


def test_pep8():
    for dir in ['pySDC/core', 'pySDC/implementations', 'pySDC/helpers', 'pySDC/tutorial', 'pySDC/projects', 'pySDC/playgrounds']:
        yield check_files, dir
