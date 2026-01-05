import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Utilities
# ---------------------------

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def ci95_from_std(std: np.ndarray, n: np.ndarray) -> np.ndarray:
    """95% CI half-width = 1.96 * std / sqrt(n)."""
    n = np.maximum(n, 1)
    se = std / np.sqrt(n)
    return 1.96 * se

def safe_std(x: pd.Series) -> float:
    # pandas std gives NaN for n=1; that's fine, we'll treat CI=0 then.
    return float(x.std(ddof=1))

def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Gains
    if "acc_fr_aligned" in df.columns and "acc_fr_unaligned" in df.columns:
        df["acc_gain_fr"] = df["acc_fr_aligned"] - df["acc_fr_unaligned"]
    if "m_fr_al" in df.columns and "m_fr_unal" in df.columns:
        df["margin_gain_fr"] = df["m_fr_al"] - df["m_fr_unal"]
    if "cos_wfrhat_wfrstar" in df.columns and "cos_raw_weng_wfrstar" in df.columns:
        df["cos_gain"] = df["cos_wfrhat_wfrstar"] - df["cos_raw_weng_wfrstar"]
    if "align_after" in df.columns and "align_before" in df.columns and "align_delta" not in df.columns:
        df["align_delta"] = df["align_after"] - df["align_before"]
    return df

def pick_relation(df: pd.DataFrame, relation: str) -> pd.DataFrame:
    if "relation" not in df.columns:
        return df
    return df[df["relation"] == relation].copy()

def aggregate_by_layer(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df.groupby(["condition", "layer"], as_index=False).agg(
        n=("seed", "count"),
        mean=(metric, "mean"),
        std=(metric, safe_std),
    )
    g["std"] = g["std"].fillna(0.0)
    g["ci95"] = ci95_from_std(g["std"].to_numpy(), g["n"].to_numpy())
    return g

def plot_metric(
    agg: pd.DataFrame,
    metric_name: str,
    title: str,
    ylabel: str,
    outpath: str,
    band: str = "ci95",
):
    plt.figure()
    for cond in sorted(agg["condition"].unique()):
        sub = agg[agg["condition"] == cond].sort_values("layer")
        x = sub["layer"].to_numpy()
        y = sub["mean"].to_numpy()
        b = sub[band].to_numpy() if band in sub.columns else sub["ci95"].to_numpy()
        line, = plt.plot(x, y, label=cond)
        plt.fill_between(x, y - b, y + b, alpha=0.2, color=line.get_color())

    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved:", outpath)

def plot_fr_margins(
    df: pd.DataFrame,
    outdir: str,
    band: str = "ci95",
):
    """
    For each condition, plot mean FR margins vs layer:
      - m_fr_unal  (using w_eng directly)
      - m_fr_al    (using W^T w_eng)
      - m_fr_star  (using w*_fr)
    With optional CI shading (computed across seeds).
    """
    needed = ["m_fr_unal", "m_fr_al", "m_fr_star"]
    for c in needed:
        if c not in df.columns:
            print(f"Skipping FR margins plot; missing column: {c}")
            return

    for cond in sorted(df["condition"].unique()):
        d = df[df["condition"] == cond].copy()

        # aggregate each margin
        def agg_margin(col: str) -> pd.DataFrame:
            g = d.groupby("layer", as_index=False).agg(
                n=("seed", "count"),
                mean=(col, "mean"),
                std=(col, safe_std),
            )
            g["std"] = g["std"].fillna(0.0)
            g["ci95"] = ci95_from_std(g["std"].to_numpy(), g["n"].to_numpy())
            return g.sort_values("layer")

        g_unal = agg_margin("m_fr_unal")
        g_al   = agg_margin("m_fr_al")
        g_star = agg_margin("m_fr_star")

        plt.figure()

        # plot with matching fill colors (matplotlib default colors)
        for g, label in [(g_unal, "FR margin (using w_eng)"),
                         (g_al,   "FR margin (using W^T w_eng)"),
                         (g_star, "FR margin (using w*_fr)")]:
            x = g["layer"].to_numpy()
            y = g["mean"].to_numpy()
            b = g[band].to_numpy() if band in g.columns else g["ci95"].to_numpy()
            line, = plt.plot(x, y, label=label)
            plt.fill_between(x, y - b, y + b, alpha=0.15, color=line.get_color())

        plt.title(f"French Margins vs Layer ({cond})")
        plt.xlabel("Layer")
        plt.ylabel("Mean margin")
        plt.legend()
        plt.tight_layout()
        outpath = os.path.join(outdir, f"fig_fr_margins_{cond}.png")
        plt.savefig(outpath)
        plt.close()
        print("Saved:", outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        action="append",
        required=True,
        help="Repeatable. Format: label:path/to/runs.jsonl (e.g., base:outputs/runs_...jsonl)",
    )
    ap.add_argument("--out_csv", type=str, default="outputs/agg_combined.csv")
    ap.add_argument("--fig_dir", type=str, default="outputs/figures")
    ap.add_argument("--band", type=str, default="ci95", choices=["ci95", "std"],
                    help="Shaded band type. Use ci95 for 95% confidence intervals.")
    ap.add_argument("--relation", type=str, default="__all__",
                    help="If runs include relation rows, select which relation to plot. Default: __all__")
    ap.add_argument("--per_relation", action="store_true",
                    help="If set and runs include relation rows, produce per-relation margin-gain plots.")
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out_csv) or ".")
    ensure_dir(args.fig_dir)

    # Load and combine runs
    all_dfs = []
    for item in args.runs:
        if ":" not in item:
            raise ValueError(f"--runs must be label:path. Got: {item}")
        label, path = item.split(":", 1)
        rows = read_jsonl(path)
        df = pd.DataFrame(rows)
        df["condition"] = label  # override/standardize
        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    df = add_derived_metrics(df)

    # If relation column exists, select relation for main plots
    df_main = pick_relation(df, args.relation)

    # Write combined CSV (all rows)
    df.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)

    # Plot the main 4 gain curves (for selected relation, usually __all__)
    plots = [
        ("align_delta", "EN→FR Procrustes Alignment Gain vs Layer", "Δ cosine (after − before)", "fig_align_delta.png"),
        ("cos_gain", "Truth-Direction Cosine Gain vs Layer", "Δ cosine (aligned − raw)", "fig_cos_gain.png"),
        ("margin_gain_fr", "French Truth Separation Margin Gain vs Layer", "Δ margin (aligned − unaligned)", "fig_margin_gain.png"),
        ("acc_gain_fr", "French Accuracy Gain vs Layer", "Δ accuracy (aligned − unaligned)", "fig_acc_gain.png"),
    ]

    for metric, title, ylabel, fname in plots:
        if metric not in df_main.columns:
            print(f"Skipping plot {fname}; missing metric: {metric}")
            continue
        agg = aggregate_by_layer(df_main, metric)
        outpath = os.path.join(args.fig_dir, fname)
        plot_metric(agg, metric, title + f" [{args.relation}]", ylabel, outpath, band=args.band)

    # Plot FR margin curves per condition (for selected relation)
    plot_fr_margins(df_main, args.fig_dir, band=args.band)

    # Optional: per-relation margin gain plots (one figure per relation)
    if args.per_relation:
        if "relation" not in df.columns:
            print("No relation column found; cannot do per-relation plots.")
            return

        rels = sorted([r for r in df["relation"].unique() if r != "__all__"])
        outdir = os.path.join(args.fig_dir, "by_relation")
        ensure_dir(outdir)

        for rel in rels:
            drel = df[df["relation"] == rel].copy()
            if "margin_gain_fr" not in drel.columns:
                continue
            agg = aggregate_by_layer(drel, "margin_gain_fr")
            outpath = os.path.join(outdir, f"fig_margin_gain_{rel}.png")
            plot_metric(
                agg,
                "margin_gain_fr",
                f"French Margin Gain vs Layer ({rel})",
                "Δ margin (aligned − unaligned)",
                outpath,
                band=args.band,
            )

        print("Per-relation plots saved to:", outdir)


if __name__ == "__main__":
    main()
