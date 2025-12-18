# -*- coding: utf-8 -*-
"""
Figure generator for econ_model_v3.py
- Produces PNGs into ./figures
- Designed to be included in LaTeX (slides or report)

Usage:
  python create_figures_v3.py --engineers 5000 --n 20000
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from econ_model_v3 import TEMPLATES, scaled_scenario, simulate

OUTDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTDIR, exist_ok=True)

def plot_net_by_template(engineers: int, n: int, seed: int):
    names = sorted(TEMPLATES.keys())
    med = []
    p10 = []
    p90 = []
    for i, t in enumerate(names):
        s = scaled_scenario(t, engineers, overrides={})
        r = simulate(s, n=n, seed=seed+i)
        med.append(r.p50/1e6)
        p10.append(r.p10/1e6)
        p90.append(r.p90/1e6)

    x = np.arange(len(names))
    plt.figure(figsize=(11, 4.8))
    plt.errorbar(x, med, yerr=[np.array(med)-np.array(p10), np.array(p90)-np.array(med)], fmt="o")
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylabel("Net value (USD, $M/year)")
    plt.title(f"AI Coding net value by industry template (engineers={engineers})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "net_value_by_template.png"), dpi=220)
    plt.close()

def plot_heatmap_adoption_uplift(template: str, engineers: int, n: int, seed: int):
    base = scaled_scenario(template, engineers, overrides={})
    adopts = np.linspace(0.15, 0.95, 17)
    mults = np.linspace(0.4, 2.2, 19)
    Z = np.zeros((len(mults), len(adopts)))

    for i, m in enumerate(mults):
        for j, a in enumerate(adopts):
            # Override only adoption and uplift multiplier via breakeven helper approach:
            # we rebuild scenario dict, edit, and simulate quickly by calling simulate on scenario copy.
            s = scaled_scenario(template, engineers, overrides={"adoption_engineer": float(a)})
            # link other roles manually (simple heuristic)
            ratio = float(a) / max(1e-6, base.engineer.adoption)
            s.qa.adoption = min(1.0, base.qa.adoption * ratio)
            s.sre.adoption = min(1.0, base.sre.adoption * ratio)
            s.pm.adoption = min(1.0, base.pm.adoption * ratio * 0.85)
            s.design.adoption = min(1.0, base.design.adoption * ratio * 0.75)

            # Multiply uplift
            def mul(tri): return (tri[0]*m, tri[1]*m, tri[2]*m)
            s.engineer.uplift = mul(base.engineer.uplift)
            s.qa.uplift = mul(base.qa.uplift)
            s.sre.uplift = mul(base.sre.uplift)
            s.pm.uplift = mul(base.pm.uplift)
            s.design.uplift = mul(base.design.uplift)

            r = simulate(s, n=n, seed=seed + i*100 + j)
            Z[i, j] = r.p50/1e6

    plt.figure(figsize=(11, 4.8))
    plt.imshow(Z, aspect="auto", origin="lower", extent=[adopts[0], adopts[-1], mults[0], mults[-1]])
    plt.colorbar(label="Median net value ($M/year)")
    plt.xlabel("Engineer adoption")
    plt.ylabel("Uplift multiplier (all roles)")
    plt.title(f"Sensitivity heatmap (template={template}, engineers={engineers})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"heatmap_{template}.png"), dpi=220)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engineers", type=int, default=3000)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--template", type=str, default="internet_bigtech", choices=sorted(TEMPLATES.keys()))
    args = ap.parse_args()

    plot_net_by_template(args.engineers, args.n, args.seed)
    plot_heatmap_adoption_uplift(args.template, args.engineers, args.n, args.seed)
    print("Saved figures to:", OUTDIR)

if __name__ == "__main__":
    main()
