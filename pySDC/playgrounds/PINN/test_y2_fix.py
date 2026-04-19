#!/usr/bin/env python
"""Quick test of y2 fixes."""

import subprocess
import sys
from pathlib import Path

def run_test():
    """Run a quick smoke test with the improved settings."""
    script_dir = Path(__file__).parent
    main_script = script_dir / "deepxde_rober_paper_simple.py"

    print("=" * 80)
    print("TESTING Y2 FIX: Quick smoke test")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  - Approach: slab_irk with IRK2 and IRK4")
    print("  - Slabs: 2")
    print("  - Steps per slab: 5 (coarse!)")
    print("  - Iterations: 500 (smoke)")
    print("  - Collocation points: 256 (with 70% in early time)")
    print("  - Loss weighting: adaptive (y2 gets ~200,000× weight)")
    print()
    print("Expected: y2 should now fit much better than before")
    print("=" * 80)
    print()

    for irk_order in [2, 4]:
        run_tag = f"y2_fix_test_irk{irk_order}"
        print(f"\nRunning IRK{irk_order}...")

        cmd = [
            sys.executable,
            str(main_script),
            "--approach", "slab_irk",
            "--irk-order", str(irk_order),
            "--num-slabs", "2",
            "--steps-per-slab", "5",
            "--iterations", "500",
            "--num-points", "256",
            "--seed", "42",
            "--run-tag", run_tag,
        ]

        result = subprocess.run(cmd, cwd=str(script_dir.parent.parent.parent))

        if result.returncode != 0:
            print(f"❌ IRK{irk_order} test failed")
            return 1

        print(f"✓ IRK{irk_order} test completed")

    print()
    print("=" * 80)
    print("Tests completed!")
    print("=" * 80)
    print()
    print("Check the plots:")
    print("  - *_y1.png: should still be good")
    print("  - *_y2.png: should now show much better fit!")
    print("  - *_y3.png: should still be good")
    print()
    return 0

if __name__ == "__main__":
    sys.exit(run_test())

