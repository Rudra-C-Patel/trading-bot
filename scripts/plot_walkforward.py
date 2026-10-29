"""Render results/walkforward.png from the saved walk-forward outputs.

Reads only the frozen files in results/momentum/ (corrected-filters run,
2026-07-10); computes nothing new beyond plotting them.

    python scripts/plot_walkforward.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "momentum"

IN_SAMPLE = "#2a78d6"    # categorical slot 1
OUT_SAMPLE = "#eb6834"   # categorical slot 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    folds = pd.read_csv(RES / "walkforward_folds.csv")
    # train CAGR of the cutoff each fold actually chose
    train = [row[f"train_cagr_{row['chosen_cutoff']:.2f}"] * 100
             for _, row in folds.iterrows()]
    test = folds["test_cagr_ann"] * 100
    eq = pd.read_csv(RES / "walkforward_oos_equity.csv", index_col=0,
                     parse_dates=True)["equity"]

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2),
                               gridspec_kw={"width_ratios": [1.15, 1]})

    x = range(len(folds))
    w = 0.38
    a.bar([i - w / 2 - 0.01 for i in x], train, w, color=IN_SAMPLE,
          label="In-sample: train CAGR of chosen cutoff")
    a.bar([i + w / 2 + 0.01 for i in x], test, w, color=OUT_SAMPLE,
          label="Out-of-sample: next 6 months (annualized)")
    a.axhline(0, color=MUTED, lw=0.8)
    for i, (tr, te) in enumerate(zip(train, test)):  # mark zero-height bars
        if tr == 0:
            a.text(i - w / 2, 0.3, "0", ha="center", color=IN_SAMPLE, fontsize=8)
        if te == 0:
            a.text(i + w / 2, 0.3, "0", ha="center", color=OUT_SAMPLE, fontsize=8)
    a.set_ylim(-15, 15)
    a.set_xticks(list(x))
    a.set_xticklabels([f"F{f}\ntop {c:.0%}" for f, c in
                       zip(folds["fold"], folds["chosen_cutoff"])])
    a.set_ylabel("Return, % per year")
    a.set_title("Each fold: what training picked vs. what it then earned",
                loc="left", fontsize=10.5)
    a.legend(frameon=False, fontsize=8.5, loc="upper left")

    b.plot(eq.index, eq / 1000, color=OUT_SAMPLE, lw=2)
    b.axhline(100, color=MUTED, lw=0.8, ls="--")
    b.set_ylabel(r"Equity, \$K (start \$100K)")
    b.set_title("Stitched out-of-sample equity, 2021-07 to 2024-12",
                loc="left", fontsize=10.5)
    b.xaxis.set_major_locator(mdates.YearLocator())
    b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    b.annotate(f"{eq.iloc[-1] / eq.iloc[0] - 1:+.1%}",
               (eq.index[-1], eq.iloc[-1] / 1000), xytext=(-4, 6),
               textcoords="offset points", ha="right", color=INK)

    for ax in (a, b):
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    out = ROOT / "results" / "walkforward.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
