#!/usr/bin/env python3
"""
Presses "Run in browser" on every page of the built website that has the button, and runs the landing page demo, in
headless Chromium, as many pages at a time as there are cores. Fails if a page does not finish, a cell raises, the
page logs a JavaScript error, the figures are not all recomputed, or the demo does not converge.
This is what readers get: Pyodide from jsDelivr, the wheels next to the pages, and run-in-browser.js.

    python -m pip install playwright && python -m playwright install chromium
    python docs/check_run_in_browser.py docs/build/html
"""

import asyncio
import functools
import http.server
import os
import sys
import threading
from pathlib import Path

from playwright.async_api import TimeoutError, async_playwright

TIMEOUT = 180_000  # ms per page; the first one also downloads about 35 MB of Pyodide
PARALLEL = os.cpu_count() or 1  # pages at a time: Pyodide runs each on one core
FINISHED = "() => /^(Done|Cell \\d+ raised|Could not start)/.test(document.querySelector('.rib-status').textContent)"


async def check(page, url):
    """Returns the problems on one page, an empty list if there are none"""
    # Only errors in our scripts: Sphinx's searchindex.js sometimes runs before searchtools.js on a busy machine
    js_errors = []
    page.on('pageerror', lambda error: 'run-in-browser' in (error.stack or '') and js_errors.append(error.message))
    await page.goto(url)
    static_figures = await page.locator('.cell_output img').count()
    await page.click('.run-in-browser')
    try:
        # run-in-browser.js answers the click at once; if it did not load, nothing ever happens
        await page.wait_for_function("() => document.querySelector('.rib-status').textContent", timeout=10_000)
        await page.wait_for_function(FINISHED, timeout=TIMEOUT)
    except TimeoutError:
        pass

    status = await page.locator('.rib-status').inner_text()
    problems = [f'JavaScript error: {error}' for error in js_errors]
    if not status.startswith('Done'):
        problems.append(f'status: {status}' if status else 'the button did nothing: did run-in-browser.js load?')
    problems += [f'cell raised:\n{error}' for error in await page.locator('pre.rib-error').all_inner_texts()]
    figures = await page.locator('.cell_output img[src^="data:"]').count()
    if figures != static_figures:
        problems.append(f'{figures} figures computed in the browser, but the page was built with {static_figures}')
    await page.close()
    return problems


async def check_unavailable(page, url):
    """A page that cannot run in the browser has to say why, next to a disabled button"""
    await page.goto(url)
    problems = []
    if not await page.locator('.run-in-browser-unavailable').is_disabled():
        problems.append('the greyed-out button is not disabled')
    if not (await page.locator('.rib-unavailable').inner_text()).strip():
        problems.append('no explanation why it does not run in the browser')
    await page.close()
    return problems


async def check_demo(page, url):
    """The landing page demo has to run SDC and MLSDC with its default setup, and plot them"""
    js_errors = []
    page.on('pageerror', lambda error: 'demo' in (error.stack or '') and js_errors.append(error.message))
    await page.goto(url)
    problems = []
    for levels in ['1', '2']:
        await page.select_option('.landing-demo select[name="levels"]', levels)
        before = await page.locator('.demo-status').inner_text()
        await page.click('.demo-run')
        try:
            await page.wait_for_function(
                "before => { const t = document.querySelector('.demo-status').textContent;"
                " return t !== before && !/^(Starting|Loading|Installing|Running)/.test(t) }",
                arg=before,
                timeout=TIMEOUT,
            )
        except TimeoutError:
            pass
        status = await page.locator('.demo-status').inner_text()
        if 'converged in' not in status:
            problems.append(f'{levels} level(s): {status}')
            break
    if await page.locator('.demo-plot').is_hidden():
        problems.append('no plot')
    problems += [f'JavaScript error: {error}' for error in js_errors]
    await page.close()
    return problems


async def run_checks(base, pages, unavailable, has_demo):
    """Runs all checks, PARALLEL pages at a time, and prints one line per page; returns whether any failed"""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        context = await browser.new_context()  # shared, so that Pyodide is downloaded once
        slots = asyncio.Semaphore(PARALLEL)

        async def run(checker, name, retry, label=''):
            async with slots:
                url = f'{base}/{name}'
                problems = await checker(await context.new_page(), url)
                if problems and retry and any(retry in p for p in problems):
                    problems = await checker(await context.new_page(), url)  # once more, in case the CDN hiccuped
            return name + label, problems

        # the first page alone, so that the others find Pyodide in the cache
        results = [await run(check, pages[0], 'status: Could not start')]
        jobs = [run(check, name, 'status: Could not start') for name in pages[1:]]
        if has_demo:
            jobs.append(run(check_demo, 'index.html', 'Could not start', ' (landing page demo)'))
        results += await asyncio.gather(*jobs)
        results += await asyncio.gather(
            *[run(check_unavailable, name, None, ' (does not run in the browser)') for name in unavailable]
        )
        await browser.close()

    failed = not has_demo
    if not has_demo:
        print('FAIL index.html has no landing page demo')
    for name, problems in results:
        tag = 'FAIL' if problems else 'n/a ' if name.endswith('(does not run in the browser)') else 'ok  '
        print(f'{tag} {name}')
        for problem in problems:
            print('     ' + problem.replace('\n', '\n     '))
        failed |= bool(problems)
    return failed


def main(site):
    html = {p.relative_to(site).as_posix(): p.read_text() for p in site.rglob('*.html')}
    pages = sorted(name for name, text in html.items() if 'run-in-browser"' in text)
    unavailable = sorted(name for name, text in html.items() if 'run-in-browser-unavailable"' in text)
    if not pages:
        sys.exit(f'No page in {site} has a "Run in browser" button: were the wheels built by docs/update_apidocs.sh?')

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):  # one line per page is enough
            pass

    handler = functools.partial(Handler, directory=site)
    class Server(http.server.ThreadingHTTPServer):
        request_queue_size = 128  # the default of 5 resets connections when several pages load at once

    server = Server(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    base = f'http://127.0.0.1:{server.server_port}'
    failed = asyncio.run(run_checks(base, pages, unavailable, 'landing-demo"' in html.get('index.html', '')))
    server.shutdown()
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main(Path(sys.argv[1] if len(sys.argv) > 1 else 'docs/build/html'))
