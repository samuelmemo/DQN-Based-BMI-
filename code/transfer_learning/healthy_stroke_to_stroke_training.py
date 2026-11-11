"""Healthy pretraining followed by leave-one-stroke-patient-out transfer."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_code = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_code))
sys.path.insert(0, str(project_code / "stroke_DQN"))

from stroke_DQN.dataset_preparation import load_dataset
from stroke_DQN.stroke_patient_training import train_patient
from transfer_learning.healthy_agent_training import train_source_agent


if __name__ == "__main__":
    window = "0p5s"
    healthy_epochs = 25
    source_epochs = 25
    target_epochs = 25
    runs = 10
    batch_size = 8

    project_results = project_code.parent / "results"
    healthy_checkpoint = project_results / "healthy_dqn_convergence" / "healthy_agent.pt"
    if not healthy_checkpoint.exists():
        raise FileNotFoundError(
            f"Train the healthy source model first: {healthy_checkpoint}"
        )

    healthy_source = torch.load(
        healthy_checkpoint, map_location="cpu", weights_only=False
    )
    healthy_state = healthy_source["model_state"]

    results = project_results / "healthy_plus_stroke_leave_one_out"
    results.mkdir(parents=True, exist_ok=True)
    (results / "settings.json").write_text(json.dumps({
        "window": window,
        "healthy_initialization": str(healthy_checkpoint),
        "source_epochs": source_epochs,
        "target_epochs": target_epochs,
        "runs": runs,
        "batch_size": batch_size,
        "initialization": "healthy model continued on the other 49 stroke patients",
    }, indent=2))

    eeg_trials, labels, patient_ids, normalization = load_dataset(
        window, dataset_type="stroke", return_normalization=True
    )
    window_dir = results / window
    window_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for target_index, target_patient in enumerate(patient_ids):
        print(f"\nTarget patient: sub-{target_patient:02d}")

        source_indices = [i for i in range(len(patient_ids)) if i != target_index]
        source_X = torch.cat([eeg_trials[i] for i in source_indices], dim=0)
        source_y = torch.cat([labels[i] for i in source_indices], dim=0)
        source_patient_ids = np.concatenate([
            np.full(len(labels[i]), patient_ids[i]) for i in source_indices
        ])
        source_normalization = [normalization[i] for i in source_indices]

        source_dir = window_dir / "source_models" / f"target-sub-{target_patient:02d}"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_filename = f"healthy_plus_stroke_source_target-sub-{target_patient:02d}.pt"
        source_checkpoint = source_dir / source_filename

        if source_checkpoint.exists():
            source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
            print(f"Using existing source model: {source_checkpoint}")
        else:
            train_source_agent(
                source_X, source_y, source_patient_ids,
                epochs=source_epochs, batch_size=batch_size,
                output_dir=source_dir, normalization=source_normalization,
                seed=7000 + int(target_patient),
                checkpoint_name=source_filename,
                initial_model_state=healthy_state,
            )
            source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)

        target_dir = window_dir / f"sub-{target_patient:02d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        rates = train_patient(
            eeg_trials[target_index], labels[target_index],
            epochs=target_epochs, runs=runs, batch_size=batch_size,
            output_dir=target_dir, normalization=normalization[target_index],
            seed_base=int(target_patient) * 1000,
            initial_model_state=source["model_state"],
        )

        completed = np.isfinite(rates).sum(axis=1)
        observed = np.ma.masked_invalid(rates)
        np.savez(
            target_dir / "curves.npz",
            success_rates=rates,
            epochs_completed=completed,
            mean=observed.mean(axis=0).filled(np.nan),
            std=observed.std(axis=0, ddof=1).filled(np.nan),
        )
        final = np.array([row[n - 1] for row, n in zip(rates, completed)])
        summary.append([
            int(target_patient),
            np.nanmean(rates[:, 0]), np.nanstd(rates[:, 0], ddof=1),
            np.nanmean(final), np.nanstd(final, ddof=1), completed.mean(),
        ])

        with (window_dir / "summary.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "patient", "first_epoch_mean", "first_epoch_std",
                "final_epoch_mean", "final_epoch_std", "mean_epochs_completed",
            ])
            writer.writerows(summary)

