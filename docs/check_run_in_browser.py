#!/usr/bin/env python3
"""
Presses "Run in browser" on every page of the built website that has the button, in headless Chromium, and fails
if a page does not finish, a cell raises, the page logs a JavaScript error, or the figures are not all recomputed.
This is what readers get: Pyodide from jsDelivr, the wheels next to the pages, and run-in-browser.js.

    python -m pip install playwright && python -m playwright install chromium
    python docs/check_run_in_browser.py docs/build/html
"""

import functools
import http.server
import sys
import threading
from pathlib import Path

from playwright.sync_api import TimeoutError, sync_playwright

TIMEOUT = 180_000  # ms per page; the first one also downloads about 35 MB of Pyodide
FINISHED = "() => /^(Done|Cell \\d+ raised|Could not start)/.test(document.querySelector('.rib-status').textContent)"


def check(page, url):
    """Returns the problems on one page, an empty list if there are none"""
    # Only errors in our scripts: Sphinx's searchindex.js sometimes runs before searchtools.js on a busy machine
    js_errors = []
    page.on('pageerror', lambda error: 'run-in-browser' in (error.stack or '') and js_errors.append(error.message))
    page.goto(url)
    static_figures = page.locator('.cell_output img').count()
    page.click('.run-in-browser')
    try:
        # run-in-browser.js answers the click at once; if it did not load, nothing ever happens
        page.wait_for_function("() => document.querySelector('.rib-status').textContent", timeout=10_000)
        page.wait_for_function(FINISHED, timeout=TIMEOUT)
    except TimeoutError:
        pass

    status = page.locator('.rib-status').inner_text()
    problems = [f'JavaScript error: {error}' for error in js_errors]
    if not status.startswith('Done'):
        problems.append(f'status: {status}' if status else 'the button did nothing: did run-in-browser.js load?')
    problems += [f'cell raised:\n{error}' for error in page.locator('pre.rib-error').all_inner_texts()]
    figures = page.locator('.cell_output img[src^="data:"]').count()
    if figures != static_figures:
        problems.append(f'{figures} figures computed in the browser, but the page was built with {static_figures}')
    page.close()
    return problems


def main(site):
    pages = sorted(p.relative_to(site).as_posix() for p in site.rglob('*.html') if 'run-in-browser"' in p.read_text())
    if not pages:
        sys.exit(f'No page in {site} has a "Run in browser" button: were the wheels built by docs/update_apidocs.sh?')

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):  # one line per page is enough
            pass

    handler = functools.partial(Handler, directory=site)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    failed = False
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context()  # shared, so that Pyodide is downloaded once
        for name in pages:
            url = f'http://127.0.0.1:{server.server_port}/{name}'
            problems = check(context.new_page(), url)
            if problems and any(p.startswith('status: Could not start') for p in problems):
                problems = check(context.new_page(), url)  # once more, in case the CDN hiccuped
            print(f'{"FAIL" if problems else "ok  "} {name}')
            for problem in problems:
                print('     ' + problem.replace('\n', '\n     '))
            failed |= bool(problems)
        browser.close()
    server.shutdown()
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main(Path(sys.argv[1] if len(sys.argv) > 1 else 'docs/build/html'))
