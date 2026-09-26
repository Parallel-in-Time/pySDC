// Runs the code cells of a tutorial page in Pyodide, off the main thread. See run-in-browser.js.
import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs';

const RUNNER = `
import ast, base64, io, json, os, sys, traceback, warnings

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning

# Pyodide's Matplotlib 3.10.8 warns about floats from its own mathtext code (FT2Image, via log-axis tick labels)
# although it passes ints. The native docs build still shows any deprecation that comes from the tutorials.
warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning)

namespace = {'__name__': '__main__'}


class CellOutput(io.TextIOBase):
    """Stands in for stdout and stderr for good, and writes to the current cell, as in Jupyter. Loggers keep the
    stream they found when they were created, e.g. a controller made in one cell and run in the next."""

    current = io.StringIO()

    def write(self, text):
        return CellOutput.current.write(text)


sys.stdout = sys.stderr = CellOutput()


def run_cell(code):
    """Run one cell like Jupyter does: show stdout, the value of a trailing expression, and all new figures"""
    CellOutput.current, error = io.StringIO(), None
    try:
        tree = ast.parse(code)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        exec(compile(tree, '<cell>', 'exec'), namespace)
        if last is not None:
            value = eval(compile(ast.Expression(last.value), '<cell>', 'eval'), namespace)
            if value is not None:
                print(repr(value))
    except Exception:
        kind, value, tb = sys.exc_info()
        error = ''.join(traceback.format_exception(kind, value, tb.tb_next))
    images = []
    for number in plt.get_fignums():
        buffer = io.BytesIO()
        plt.figure(number).savefig(buffer, format='png', bbox_inches='tight')
        images.append(base64.b64encode(buffer.getvalue()).decode())
    plt.close('all')
    return json.dumps({'stdout': CellOutput.current.getvalue(), 'error': error, 'images': images})
`;

let pyodide;

onmessage = async ({ data }) => {
  if (data.type === 'init') {
    try {
      pyodide = await loadPyodide();
      postMessage({ type: 'status', text: 'Loading NumPy, SciPy and Matplotlib…' });
      await pyodide.loadPackage(['numpy', 'scipy', 'matplotlib', 'micropip']);
      postMessage({ type: 'status', text: 'Installing pySDC…' });
      pyodide.globals.set('wheels', data.wheels);
      // The wheels are built from the same commit as the page, and pySDC's other dependencies are loaded above
      await pyodide.runPythonAsync('import micropip\nfor w in wheels: await micropip.install(w, deps=False)');
      await pyodide.runPythonAsync(RUNNER);
      postMessage({ type: 'ready' });
    } catch (error) {
      postMessage({ type: 'failed', text: String(error) });
    }
  } else if (data.type === 'run') {
    pyodide.globals.set('code', data.code);
    postMessage({ type: 'result', id: data.id, ...JSON.parse(pyodide.runPython('run_cell(code)')) });
  }
};
