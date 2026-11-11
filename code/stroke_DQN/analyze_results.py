"""Summarize saved training curves. Rates in the CSV are percentages."""

import csv
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save the figure without opening a window.
import matplotlib.pyplot as plt
import numpy as np


def analyze(results):
    rows = []
    overall = []
    output = results / "analysis" / "figures_and_tables"
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for window, label in [("1p0s", "1 second"), ("0p5s", "0.5 seconds")]:
        patient_curves, first_means, last_means = [], [], []
        for path in sorted((results / window).glob("sub-*/curves.npz")):
            with np.load(path) as data:
                rates = data["success_rates"] * 100
            if rates.ndim != 2 or rates.shape[0] < 2 or rates.shape[1] < 3:
                raise ValueError(f"Expected multiple runs and at least 3 epochs: {path}")
            # One first-epoch score and one final-three-epoch average per run.
            first = rates[:, 0]
            completed = np.isfinite(rates).sum(axis=1)
            if np.any(completed == 0):
                raise ValueError(f"Run with no completed epochs: {path}")
            # If a run stopped before epoch 3, average its available epochs.
            last = np.array([row[np.isfinite(row)][-3:].mean() for row in rates])
            rows.append([window, path.parent.name, len(rates), completed.mean(),
                         first.mean(), first.std(ddof=1), last.mean(), last.std(ddof=1),
                         last.mean() - first.mean()])
            first_means.append(first.mean())
            last_means.append(last.mean())
            patient_curves.append(np.ma.masked_invalid(rates).mean(axis=0).filled(np.nan))

        if not patient_curves:
            print(f"{label}: no saved curves yet")
            continue
        # Average runs within patients first, then give every patient equal weight.
        curves = np.stack(patient_curves)
        observed = np.ma.masked_invalid(curves)
        mean = observed.mean(axis=0).filled(np.nan)
        sd = observed.std(axis=0, ddof=1).filled(np.nan) if len(curves) > 1 else np.zeros(curves.shape[1])
        epochs = np.arange(1, len(mean) + 1)
        ax.plot(epochs, mean, label=label)
        ax.fill_between(epochs, mean - sd, mean + sd, alpha=0.15)
        # Separate black-and-gray curves, like the reference (no TL condition here).
        curve_fig, curve_ax = plt.subplots(figsize=(8, 4.8), facecolor="#f4f4f4")
        curve_ax.plot(epochs, mean, color="black", linewidth=2, label="Without TL")
        curve_ax.fill_between(epochs, np.clip(mean - sd, 0, 100),
                              np.clip(mean + sd, 0, 100), color="gray", alpha=0.3)
        curve_ax.set(xlabel="Epoch", ylabel="Success Rate (%)", title=label,
                     xlim=(1, len(mean)), ylim=(40, 101))
        curve_ax.grid(alpha=0.25)
        curve_ax.legend(loc="lower right", framealpha=1, fancybox=False)
        curve_ax.tick_params(labelsize=12)
        curve_fig.tight_layout()
        curve_fig.savefig(output / f"learning_curve_{window}.png", dpi=200)
        plt.close(curve_fig)
        overall.append([window, len(curves), np.mean(first_means),
                        np.std(first_means, ddof=1)])
        print(f"{label}: {len(curves)} patients")
        print(f"  First epoch: {np.mean(first_means):.2f}%")
        print(f"  Final up to 3 completed epochs (average): {np.mean(last_means):.2f}%")
        print(f"  Improvement: {np.mean(last_means) - np.mean(first_means):.2f} percentage points")

    if not rows:
        plt.close(fig)
        raise SystemExit("No curves found. Check the results directory.")
    with (output / "patient_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window", "patient", "runs", "mean_epochs_completed", "first_epoch_mean_pct",
                         "first_epoch_std_pct", "last_up_to_3_mean_pct", "last_up_to_3_std_pct",
                         "improvement_percentage_points"])
        writer.writerows(rows)

    # Group the two windows for each subject; error bars are SD across runs.
    subjects = sorted({row[1] for row in rows})
    bar_fig, bar_ax = plt.subplots(figsize=(18, 6))
    positions = np.arange(len(subjects))
    for offset, window, label, color in [(-0.2, "0p5s", "0.5 seconds", "#1479b8"),
                                        (0.2, "1p0s", "1 second", "#d95300")]:
        lookup = {row[1]: row for row in rows if row[0] == window}
        means = [lookup[s][4] if s in lookup else np.nan for s in subjects]
        errors = [lookup[s][5] if s in lookup else np.nan for s in subjects]
        bar_ax.bar(positions + offset, means, width=0.4, yerr=errors, capsize=2,
                   color=color, edgecolor="black", linewidth=0.5, label=label)
    bar_ax.set(xlabel="Subject", ylabel="First-epoch success rate (%)", ylim=(0, 100),
               title="Without transfer learning; error bars = SD across runs")
    bar_ax.set_xticks(positions, subjects, rotation=60, ha="right")
    bar_ax.set_axisbelow(True)
    bar_ax.grid(axis="y", alpha=0.25)
    bar_ax.legend()
    bar_fig.tight_layout()
    bar_fig.savefig(output / "first_epoch_by_subject.png", dpi=200)
    plt.close(bar_fig)
    with (output / "overall_first_epoch.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window", "subjects", "mean_first_epoch_pct", "std_across_subject_means_pct"])
        writer.writerows(overall)

    ax.set(xlabel="Epoch", ylabel="Training success (%)",
                title="Observed learning curves; shading = SD across patients")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "training_summary.png", dpi=180)
    plt.close(fig)
    print(f"Saved CSV and figure to {output}")
    print("These are training results, not held-out test accuracy.")
    print("After early stopping, later epochs average only runs/patients still training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results" / "stroke_dqn_convergence_buffer1000_shuffle")
    analyze(parser.parse_args().results)
