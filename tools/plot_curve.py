# -*- coding: utf-8 -*-
"""
Plot BLEU-1 vs. SNR from a quick_eval.log (or evaluate output) into a PNG.

Reads lines of the form
    SNR <int> dB | BLEU-1 = <float>
so the figure always matches whatever was actually measured.
"""
import re
import sys
import argparse
import matplotlib
matplotlib.use("Agg")  # headless: write to file, no display needed
import matplotlib.pyplot as plt

LINE = re.compile(r"SNR\s+(\d+)\s*dB\s*\|\s*BLEU-1\s*=\s*([0-9.]+)")


def read_curve(path):
    snr, bleu = [], []
    with open(path) as f:
        for line in f:
            m = LINE.search(line)
            if m:
                snr.append(int(m.group(1)))
                bleu.append(float(m.group(2)))
    # sort by SNR just in case the log is out of order
    pairs = sorted(zip(snr, bleu))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="data/quick_eval.log")
    parser.add_argument("--out", default="figures/bleu_vs_snr.png")
    parser.add_argument("--channel", default="Rayleigh")
    parser.add_argument("--note", default="epoch-7 checkpoint · Europarl EN · 2000-sentence subset")
    args = parser.parse_args()

    snr, bleu = read_curve(args.log)
    if not snr:
        sys.exit("no 'SNR .. dB | BLEU-1 = ..' lines found in " + args.log)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.plot(snr, bleu, "-o", color="#1f6feb", lw=2, ms=8,
            markerfacecolor="white", markeredgewidth=2, zorder=3)

    # value labels above each point
    for x, y in zip(snr, bleu):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color="#1f6feb")

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("BLEU-1 score", fontsize=12)
    ax.set_title("DeepSC Semantic Communication — BLEU-1 vs. SNR",
                 fontsize=14, fontweight="bold", pad=12)

    ax.set_xticks(snr)
    ax.set_ylim(min(bleu) - 0.05, max(bleu) + 0.08)
    ax.grid(True, ls="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # meta info box
    ax.text(0.02, 0.02, "channel: {}\n{}".format(args.channel, args.note),
            transform=ax.transAxes, fontsize=9, color="#555",
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f6f8fa", ec="#d0d7de"))

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("saved:", args.out)
    print("SNR  :", snr)
    print("BLEU :", [round(b, 4) for b in bleu])


if __name__ == "__main__":
    main()
