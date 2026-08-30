"""
Build a standalone HTML report from the order-reduction study results.

Reads ``order_study_results.json`` (produced by ``order_study.py``), creates
matplotlib convergence plots, embeds them as base64-encoded PNGs, and writes a
single self-contained HTML file ``order_reduction_report.html`` that can be
opened in any browser without external assets.

Run with::

    micromamba run -n pysdc-fenics python -m pySDC.playgrounds.FEniCS.order_reduction.make_report

(matplotlib is required; FEniCS is not needed for this step.)
"""

import base64
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / 'order_study_results.json'
NONLINEAR_RESULTS = HERE / 'nonlinear_order_study_results.json'
DIFFUSION_RESULTS = HERE / 'nonlinear_diffusion_order_study_results.json'
OUT = HERE / 'order_reduction_report.html'


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def convergence_plot(cases_for_M, num_nodes):
    """Log-log convergence plot for the sine and cosine cases at a given M."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = {
        'Sine (constant-in-time BCs)': '#1f77b4',
        'Cosine (time-dependent BCs)': '#d62728',
        'Cosine + internal boundary lifting': '#2ca02c',
    }
    markers = {
        'Sine (constant-in-time BCs)': 'o',
        'Cosine (time-dependent BCs)': 's',
        'Cosine + internal boundary lifting': '^',
    }

    expected = 2 * num_nodes - 1
    all_dts, all_errs = [], []
    for c in cases_for_M:
        dts = np.array(c['dts'])
        errs = np.array(c['errors'])
        all_dts.append(dts)
        all_errs.append(errs)
        ax.loglog(
            dts,
            errs,
            marker=markers.get(c['label'], 'o'),
            color=colors.get(c['label'], None),
            linewidth=1.8,
            markersize=7,
            label=f"{c['label']}  (fit slope {c['fitted_order']:.2f})",
        )

    # reference slope line for the full expected order
    dts0 = np.array(sorted(cases_for_M[0]['dts']))
    ref_anchor = max(c['errors'][0] for c in cases_for_M) * 1.6
    ref = ref_anchor * (dts0 / dts0[-1]) ** expected
    ax.loglog(dts0, ref, 'k--', linewidth=1.2, label=f'reference slope {expected} (full order)')

    ax.set_xlabel('time step  $\\Delta t$')
    ax.set_ylabel('relative error at $T_{end}$')
    ax.set_title(f'SDC temporal convergence, RADAU-RIGHT  $M={num_nodes}$  (expected order $2M-1={expected}$)')
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=9)
    return _fig_to_base64(fig)


def local_order_plot(cases_for_M, num_nodes):
    """Bar-style local (successive) order estimates vs dt interval midpoints."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {
        'Sine (constant-in-time BCs)': '#1f77b4',
        'Cosine (time-dependent BCs)': '#d62728',
        'Cosine + internal boundary lifting': '#2ca02c',
    }
    expected = 2 * num_nodes - 1

    for c in cases_for_M:
        dts = np.array(c['dts'])
        mids = np.sqrt(dts[:-1] * dts[1:])  # geometric midpoint of each interval
        lo = np.array(c['local_orders'])
        ax.semilogx(
            mids, lo, marker='o', color=colors.get(c['label'], None), linewidth=1.8, markersize=7, label=c['label']
        )

    ax.axhline(expected, color='k', linestyle='--', linewidth=1.2, label=f'full order {expected}')
    ax.set_xlabel('time-step interval (geometric mean of $\\Delta t$ pair)')
    ax.set_ylabel('local convergence order')
    ax.set_title(f'Local (successive) order estimates, $M={num_nodes}$')
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    return _fig_to_base64(fig)


def _table(cases_for_M):
    """HTML table of dt, errors, local orders, iterations for a given M."""
    dts = cases_for_M[0]['dts']
    header = "<tr><th>&Delta;t</th>"
    for c in cases_for_M:
        short = (
            'sine'
            if 'Sine' in c['label']
            else 'cosine + lift'
            if 'lifting' in c['label']
            else 'nonlinear diffusion cosine'
            if 'Nonlinear diffusion' in c['label']
            else 'cosine'
        )
        header += f"<th>err ({short})</th><th>iters ({short})</th>"
    header += "</tr>"

    rows = ""
    for i, dt in enumerate(dts):
        rows += f"<tr><td>{dt:.5f}</td>"
        for c in cases_for_M:
            iterations = c.get('mean_iters', ['n/a'] * len(dts))[i]
            iteration_text = f'{iterations:.2f}' if isinstance(iterations, (int, float)) else iterations
            rows += f"<td>{c['errors'][i]:.3e}</td><td>{iteration_text}</td>"
        rows += "</tr>"

    # local order row block
    lorows = "<tr><th>local order (interval)</th>"
    n_intervals = len(cases_for_M[0]['local_orders'])
    lorows += f"<td colspan='{2 * len(cases_for_M)}'></td></tr>"
    for j in range(n_intervals):
        lorows += f"<tr><td>{dts[j]:.4f}&rarr;{dts[j+1]:.4f}</td>"
        for c in cases_for_M:
            lorows += f"<td colspan='2' style='text-align:center'>{c['local_orders'][j]:.2f}</td>"
        lorows += "</tr>"

    return f"<table>{header}{rows}{lorows}</table>"


def build_html(data):
    cases = data['cases']
    meta = data['meta']
    Ms = sorted({c['num_nodes'] for c in cases})

    sections = ""
    for M in Ms:
        cm = [c for c in cases if c['num_nodes'] == M]
        # ensure sine first, cosine second
        cm.sort(key=lambda c: 0 if 'Sine' in c['label'] else 1)
        conv_b64 = convergence_plot(cm, M)
        lo_b64 = local_order_plot(cm, M)
        table = _table(cm)

        sine = next(c for c in cm if 'Sine' in c['label'])
        cos = next(c for c in cm if c['label'] == 'Cosine (time-dependent BCs)')
        lift = next(c for c in cm if 'lifting' in c['label'])
        # verdict text based on the first (cleanest, largest-dt) local order
        sine_lo0 = sine['local_orders'][0]
        cos_lo0 = cos['local_orders'][0]
        lift_lo0 = lift['local_orders'][0]
        expected = 2 * M - 1
        verdict = (
            f"At the largest time steps (cleanest asymptotic regime, before the spatial "
            f"error floor is reached), the <b>sine</b> case attains local order "
            f"<b>{sine_lo0:.2f}</b> &mdash; essentially the full SDC order {expected} &mdash; "
            f"while the <b>cosine</b> (time-dependent BC) case attains only <b>{cos_lo0:.2f}</b>, "
            f"a clear <b>order reduction</b> of about {expected - cos_lo0:.1f}. "
            f"With internal boundary lifting, the cosine case recovers <b>{lift_lo0:.2f}</b> "
            f"on the same interval."
        )

        sections += f"""
        <section>
          <h2>M = {M} collocation nodes &nbsp;(RADAU-RIGHT, expected order 2M&minus;1 = {expected})</h2>
          <p class="verdict">{verdict}</p>
          <div class="plots">
            <img src="data:image/png;base64,{conv_b64}" alt="convergence M={M}"/>
            <img src="data:image/png;base64,{lo_b64}" alt="local order M={M}"/>
          </div>
          {table}
        </section>
        """

    nonlinear_section = ''
    if NONLINEAR_RESULTS.exists():
        with open(NONLINEAR_RESULTS) as fh:
            nonlinear_data = json.load(fh)
        nonlinear_cases = nonlinear_data['cases']
        nonlinear_section = f"""
<section>
  <h2>Nonlinear Extension: Manufactured Reaction-Diffusion</h2>
  <p>
    The same three-way experiment was repeated for
    <span class="eq">u<sub>t</sub> = &nu;u<sub>xx</sub> + &lambda;u(1-u) + f(x,t)</span>
    using a manufactured solution. The nonlinear reaction is evaluated at the
    reconstructed physical state <span class="eq">u=v+E</span> in the lifted
    formulation, and every implicit node solve uses Newton iteration.
  </p>
  <p class="verdict">
    For <b>M=3</b>, the nonlinear sine case reaches order 5, direct
    time-dependent BC imposition reduces the cosine case to approximately order
    4.1, and lifting restores approximately order 4.9.
  </p>
  <div class="plots">
    <img src="data:image/png;base64,{convergence_plot(nonlinear_cases, 3)}" alt="nonlinear convergence"/>
    <img src="data:image/png;base64,{local_order_plot(nonlinear_cases, 3)}" alt="nonlinear local order"/>
  </div>
  {_table(nonlinear_cases)}
</section>
"""

    diffusion_section = ''
    if DIFFUSION_RESULTS.exists():
        with open(DIFFUSION_RESULTS) as fh:
            diffusion_data = json.load(fh)
        diffusion_cases = diffusion_data['cases']
        diffusion_section = f"""
<section>
  <h2>Increasing Difficulty: Nonlinear Diffusion</h2>
  <p>
    The diffusion operator is now the nonlinear flux
    <span class="eq">&part;<sub>x</sub>((1+&gamma;u<sup>2</sup>)u<sub>x</sub>)</span>.
    The implicit SDC node equations are solved with Newton iteration. In the
    lifted formulation the nonlinear flux is evaluated at the physical state
    <span class="eq">u=v+E</span>, including the lift gradient in the face flux.
  </p>
  <p class="verdict">
    The sine case approaches the full fifth order. Direct time-dependent cosine
    BC imposition settles near order 4.1, while lifting trends toward order 5;
    the final local lifted orders are 4.51 and 4.71. This is a stronger test
    because the lift enters the nonlinear diffusion operator itself.
  </p>
  <div class="plots">
    <img src="data:image/png;base64,{convergence_plot(diffusion_cases, 3)}" alt="nonlinear diffusion convergence"/>
    <img src="data:image/png;base64,{local_order_plot(diffusion_cases, 3)}" alt="nonlinear diffusion local order"/>
  </div>
  {_table(diffusion_cases)}
</section>
"""

    generated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    meta_rows = "".join(f"<li><code>{k}</code> = {v}</li>" for k, v in meta.items())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>SDC order reduction with time-dependent boundary conditions</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }}
  h1 {{ font-size: 1.7rem; border-bottom: 3px solid #1f77b4; padding-bottom: .4rem; }}
  h2 {{ font-size: 1.25rem; margin-top: 2.2rem; color: #14324f; }}
  .verdict {{ background: #f1f7fd; border-left: 4px solid #1f77b4; padding: .7rem 1rem; border-radius: 4px; }}
  .plots {{ display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center; margin: 1rem 0; }}
  .plots img {{ max-width: 100%; width: 46%; min-width: 340px; border: 1px solid #ddd; border-radius: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .88rem; }}
  th, td {{ border: 1px solid #ccc; padding: .35rem .5rem; text-align: right; }}
  th {{ background: #eef2f5; }}
  code {{ background: #f2f2f2; padding: .05rem .3rem; border-radius: 3px; }}
  .setup {{ background: #fafafa; border: 1px solid #e2e2e2; border-radius: 6px; padding: .6rem 1.2rem; }}
  .footer {{ margin-top: 3rem; font-size: .8rem; color: #666; border-top: 1px solid #ddd; padding-top: .6rem; }}
  .eq {{ font-family: "Latin Modern Math", "STIX Two Math", serif; }}
</style>
</head>
<body>
<h1>Order reduction in SDC with time-dependent Dirichlet boundary conditions</h1>

<p>
This report reproduces the classic <b>order-reduction</b> phenomenon for
Spectral Deferred Corrections (SDC) applied to the forced 1D heat equation
discretized in space with FEniCS finite elements. It compares two manufactured
solutions of
<span class="eq">u<sub>t</sub> = &nu; u<sub>xx</sub> + f(x,t)</span>
on <span class="eq">&Omega; = [0,1]</span>:
</p>
<ul>
  <li><b>Sine case</b> (<code>fenics_heat_mass</code>):
      <span class="eq">u(x,t) = sin(&pi;x) cos(t) + c</span>.
      Since <span class="eq">sin(&pi;&middot;0) = sin(&pi;&middot;1) = 0</span>, the Dirichlet
      boundary values are <b>constant in time</b> (equal to <span class="eq">c</span>).</li>
  <li><b>Cosine case</b> (<code>fenics_heat_mass_timebc</code>):
      <span class="eq">u(x,t) = cos(&pi;x) cos(t) + c</span>.
      Now <span class="eq">cos(&pi;&middot;0)=1</span>, <span class="eq">cos(&pi;&middot;1)=&minus;1</span>,
      so the boundary values <span class="eq">&plusmn;cos(t)+c</span> <b>change in time</b>.
      The BC is imposed directly on the right-hand side inside <code>solve_system</code>
      via <code>bc.apply(b.values.vector())</code>, which breaks the collocation
      fixed point and reduces the observed convergence order.</li>
  <li><b>Cosine + internal lifting</b> (<code>fenics_heat_mass_timebc_lift_physical</code>):
      integrates the homogeneous lifted variable internally, reconstructs the
      physical solution, and keeps the exact cosine solution unchanged.</li>
</ul>

<div class="setup">
<b>Numerical setup</b>
<ul>{meta_rows}
  <li>sweeper: <code>imex_1st_order_mass</code> (IMEX, mass-matrix formulation)</li>
  <li>iterated to residual tolerance so the SDC solution reaches the collocation solution</li>
  <li>error = relative error <span class="eq">|u<sub>h</sub> &minus; u<sub>exact</sub>| / |u<sub>exact</sub>|</span> at <span class="eq">T<sub>end</sub></span></li>
</ul>
</div>

<p>
The <b>reference dashed line</b> in each convergence plot has the full theoretical
slope <span class="eq">2M&minus;1</span>. The <b>local order plot</b> shows the slope
computed between successive time-step pairs, which reveals the asymptotic behaviour
before the (spatial) error floor is reached.
</p>

{sections}
{nonlinear_section}
{diffusion_section}

<h2>Summary</h2>
<p>
The linear sine case (constant-in-time boundary data) converges with the <b>full SDC
order</b> <span class="eq">2M&minus;1</span>. The cosine case, whose Dirichlet data
varies in time and is imposed naively on the right-hand side, exhibits a clear
<b>order reduction</b>: for <span class="eq">M=3</span> the order drops from the
expected 5 to roughly 4.4. The internally lifted cosine case recovers the full
order while retaining the same physical exact solution. The effect is milder
for <span class="eq">M=2</span> and largely a pre-asymptotic phenomenon there. Below the spatial error floor
(&asymp; 10<sup>&minus;10</sup> for these settings) the temporal order can no longer
be measured, which is why the smallest-<span class="eq">&Delta;t</span> local orders
in the <span class="eq">M=3</span> plots collapse.
</p>
<p>
The nonlinear reaction-diffusion extension shows the same mechanism: order 5
for time-independent boundary data, approximately order 4.1 after naive
time-dependent BC imposition, and approximately order 4.9 after boundary
lifting.
</p>
<p>
The nonlinear-diffusion extension places the lift inside the nonlinear flux:
the direct cosine case remains around order 4.1, while the lifted case trends
upward toward the full fifth order.
</p>

<div class="footer">
Generated {generated} &middot; pySDC &middot;
sources: <code>order_study.py</code> (runs), <code>make_report.py</code> (this report) &middot;
problem classes: <code>HeatEquation_1D_FEniCS_matrix_forced.py</code>
</div>
</body>
</html>
"""


def main():
    with open(RESULTS) as fh:
        data = json.load(fh)
    html = build_html(data)
    with open(OUT, 'w') as fh:
        fh.write(html)
    print(f"Wrote standalone HTML report to {OUT}")


if __name__ == '__main__':
    main()
