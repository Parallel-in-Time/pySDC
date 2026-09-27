// "Run in browser" for tutorial pages: runs their code cells in Pyodide and makes them editable.
// The button comes from _templates/notebook-links.html, only on pages listed in BROWSER_PAGES in conf.py.

let worker, nextId = 0;
const pending = new Map();

export function startPython(button, status) {
  worker = new Worker(new URL('run-in-browser-worker.js', import.meta.url), { type: 'module' });
  const wheels = button.dataset.wheels.split(' ').map(w => new URL(button.dataset.wheelsUrl + w, location.href).href);
  return new Promise((resolve, reject) => {
    worker.onmessage = ({ data }) => {
      if (data.type === 'status') status(data.text);
      else if (data.type === 'ready') resolve();
      else if (data.type === 'failed') reject(new Error(data.text));
      else if (data.type === 'result') { pending.get(data.id)(data); pending.delete(data.id); }
    };
    worker.postMessage({ type: 'init', wheels });
  });
}

export function runCode(code) {
  const id = nextId++;
  return new Promise(resolve => { pending.set(id, resolve); worker.postMessage({ type: 'run', id, code }); });
}

function showOutput(cell, { stdout, error, images }) {
  let output = cell.querySelector(':scope > .cell_output');
  if (!output) {
    output = document.createElement('div');
    output.className = 'cell_output docutils container';
    cell.append(output);
  }
  output.replaceChildren();
  const text = (content, className) => {
    const pre = document.createElement('pre');
    pre.className = className;
    pre.textContent = content;
    output.append(pre);
  };
  if (stdout) text(stdout, 'rib-stdout');
  for (const png of images) {
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${png}`;
    img.alt = 'Figure computed in your browser';
    output.append(img);
  }
  if (error) text(error, 'rib-error');
}

async function runCell(cell) {
  cell.classList.add('rib-running');
  const result = await runCode(cell.querySelector('.cell_input pre').textContent);
  cell.classList.remove('rib-running');
  showOutput(cell, result);
  return !result.error;
}

function makeEditable(cell, queue) {
  const pre = cell.querySelector('.cell_input pre');
  pre.contentEditable = 'plaintext-only';
  pre.spellcheck = false;
  pre.addEventListener('keydown', event => {
    if (event.key === 'Enter' && event.shiftKey) {
      event.preventDefault();
      queue(() => runCell(cell));
    } else if (event.key === 'Tab') {
      event.preventDefault();
      document.execCommand('insertText', false, '    ');
    }
  });
  const run = document.createElement('button');
  run.className = 'rib-cell-run';
  run.title = 'Run this cell (Shift+Enter)';
  run.textContent = '▶';
  run.addEventListener('click', () => queue(() => runCell(cell)));
  cell.querySelector('.cell_input').prepend(run);
}

document.addEventListener('DOMContentLoaded', () => {
  const button = document.querySelector('.run-in-browser');
  if (!button) return;
  const statusLine = document.querySelector('.rib-status');
  const status = text => (statusLine.textContent = text);
  const cells = [...document.querySelectorAll('article div.cell')].filter(cell => cell.querySelector('.cell_input pre'));

  // One cell at a time, in the order they were asked for, as in a notebook
  let chain = Promise.resolve();
  const queue = task => (chain = chain.then(task));

  const runAll = () => queue(async () => {
    button.disabled = true;
    for (const [i, cell] of cells.entries()) {
      status(`Running cell ${i + 1} of ${cells.length}…`);
      if (!(await runCell(cell))) {
        status(`Cell ${i + 1} raised an error. Fix it and press ▶ or Shift+Enter, or run everything again.`);
        cell.scrollIntoView({ behavior: 'smooth', block: 'center' });
        button.disabled = false;
        return;
      }
    }
    status('Done. Edit any cell and rerun it with ▶ or Shift+Enter. Cells share their variables, as in a notebook.');
    button.disabled = false;
  });

  button.addEventListener('click', async () => {
    if (worker) return runAll();
    button.disabled = true;
    status('Starting Python in your browser (the first visit downloads about 35 MB)…');
    try {
      await startPython(button, status);
    } catch (error) {
      status(`Could not start Python: ${error.message}`);
      button.disabled = false;
      worker.terminate();
      worker = undefined;
      return;
    }
    cells.forEach(cell => makeEditable(cell, queue));
    document.body.classList.add('rib-live');
    button.innerHTML = '<i class="fa-solid fa-rotate-right"></i> Run all again';
    runAll();
  });
});
