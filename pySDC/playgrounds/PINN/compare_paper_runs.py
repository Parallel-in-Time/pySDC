#!/usr/bin/env python3
"""Compare paper-mode QSSA vs non-QSSA PINN results.

Reads the metrics files written by both runs and prints a side-by-side
table, then writes a markdown summary to data/paper_comparison_summary.md.
"""
from __future__ import annotations

from pathlib import Path


def read_metrics(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="ascii") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def fmt(val: str, decimals: int = 3) -> str:
    try:
        return f"{float(val):.{decimals}e}"
    except ValueError:
        return val


def main() -> None:
    data_dir = Path(__file__).with_name("data")

    qssa_path    = data_dir / "stiff_pinn_robertson_qssa_paper200k_metrics.txt"
    noqussa_path = data_dir / "deepxde_rober_paper_simple_paper200k_metrics.txt"

    missing = [p for p in (qssa_path, noqussa_path) if not p.exists()]
    if missing:
        print("ERROR: the following metrics files are missing:")
        for p in missing:
            print(f"  {p}")
        return

    q  = read_metrics(qssa_path)
    nq = read_metrics(noqussa_path)

    # Normalise key names: QSSA uses rel_l2_yi, non-QSSA uses rmse_yi
    def get_error(m: dict, i: int) -> str:
        for key in (f"rel_l2_y{i}", f"rmse_y{i}"):
            if key in m:
                return fmt(m[key])
        return "N/A"

    rows = [
        ("Approach",         "QSSA (stiff_pinn_robertson_qssa)",  "non-QSSA (deepxde_rober_paper_simple)"),
        ("Epochs/iterations", q.get("epochs", "?"),               nq.get("iterations", "?")),
        ("Train points",      q.get("n_grid_train", "?"),         nq.get("num_points", "?")),
        ("Batch size",        q.get("batch_size", "?"),           nq.get("batch_size", "?")),
        ("Learning rate",     q.get("learning_rate", "?"),        nq.get("lr", "?")),
        ("t range",           f"[{q.get('t_min','?')}, {q.get('t_max','?')}]",
                              f"[{nq.get('t_min','?')}, {nq.get('t_max','?')}]"),
        ("Error y1",          get_error(q, 1),                    get_error(nq, 1)),
        ("Error y2",          get_error(q, 2),                    get_error(nq, 2)),
        ("Error y3",          get_error(q, 3),                    get_error(nq, 3)),
        ("Final train loss",  fmt(q.get("final_train_loss", q.get("train_loss","N/A"))),
                              "N/A"),
    ]

    # Add mass drift if present
    if "mass_drift" in q:
        rows.append(("Mass drift |Σy-1|", fmt(q["mass_drift"]), "N/A"))

    # Console output
    col_w = [max(len(r[c]) for r in rows) for c in range(3)]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    print(sep)
    for i, row in enumerate(rows):
        line = "| " + " | ".join(cell.ljust(w) for cell, w in zip(row, col_w)) + " |"
        print(line)
        if i == 0:
            print(sep)
    print(sep)

    # Markdown summary
    md_lines = ["# Paper-mode Comparison: QSSA vs non-QSSA", ""]
    md_lines.append("| " + " | ".join(rows[0]) + " |")
    md_lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")
    for row in rows[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.append("")
    md_lines.append(f"QSSA metrics from: `{qssa_path.name}`")
    md_lines.append(f"non-QSSA metrics from: `{noqussa_path.name}`")

    summary_path = data_dir / "paper_comparison_summary.md"
    summary_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()

