// The demo on the landing page: SDC or MLSDC for one step of the 1D Allen-Cahn equation, run in the browser with
// the same Pyodide worker as "Run in browser". The Python side is landing_demo.py; the form comes from conf.py.
// Import run-in-browser.js by the URL the page loaded it with, ?v= included: the same module, not a second copy, and
// never a stale one from the cache after an update
const { startPython, runCode } = await import(document.querySelector('script[src*="run-in-browser.js"]').src);

function setUp() {
  const demo = document.querySelector('.landing-demo');
  if (!demo) return;
  const form = demo.querySelector('form');
  const run = demo.querySelector('.demo-run');
  const clear = demo.querySelector('.demo-clear');
  const statusLine = demo.querySelector('.demo-status');
  const plot = demo.querySelector('.demo-plot');
  const status = text => (statusLine.textContent = text);
  let started = false;

  const show = ({ stdout, error, images }) => {
    if (images.length) plot.src = `data:image/png;base64,${images[images.length - 1]}`;
    status(error ? error.trim().split('\n').pop() : stdout.trim());
    return !error;
  };

  // MIN-SR-FLEX changes QΔ from sweep to sweep, which pySDC only allows for single-level SDC
  const flex = form.QI.querySelector('option[value="MIN-SR-FLEX"]');
  const syncFlex = () => {
    flex.disabled = form.levels.value === '2';
    if (flex.disabled && form.QI.value === 'MIN-SR-FLEX') form.QI.value = 'MIN-SR-S';
  };
  form.levels.addEventListener('change', syncFlex);
  syncFlex();

  form.addEventListener('submit', async event => {
    event.preventDefault();
    run.disabled = clear.disabled = true;
    try {
      if (!started) {
        status('Starting Python in your browser (the first visit downloads about 35 MB)…');
        await startPython(demo, status);
        const source = await (await fetch(new URL('landing_demo.py', import.meta.url), { cache: 'reload' })).text();
        if (!show(await runCode(source))) return;
        started = true;
      }
      const f = form.elements;
      status('Running…');
      show(await runCode(
        `show(dt=${f.dt.value}, QI='${f.QI.value}', num_nodes=${f.num_nodes.value}, levels=${f.levels.value}, eps=${f.eps.value})`
      ));
      plot.hidden = false;
    } catch (error) {
      status(`Could not start Python: ${error.message}`);
    } finally {
      run.disabled = false;
      clear.disabled = !started;
    }
  });

  clear.addEventListener('click', async () => {
    await runCode('clear()');
    plot.hidden = true;
    status('Cleared. Choose a setup and press Run.');
  });
}

// the await above may have outlasted DOMContentLoaded
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', setUp);
else setUp();
