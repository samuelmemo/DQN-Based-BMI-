"""Fine-tune a healthy pretrained EEGNet separately for each stroke patient."""

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


if __name__ == "__main__":
    window = "0p5s"
    epochs = 25
    runs = 10
    batch_size = 8

    healthy_checkpoint = (
        project_code.parent / "results" / "healthy_dqn_convergence" / "healthy_agent.pt"
    )
    if not healthy_checkpoint.exists():
        raise FileNotFoundError(
            f"Train the healthy source model first: {healthy_checkpoint}"
        )

    source = torch.load(healthy_checkpoint, map_location="cpu", weights_only=False)
    pretrained_state = source["model_state"]
    results = project_code.parent / "results" / "healthy_to_stroke_convergence"
    results.mkdir(parents=True, exist_ok=True)
    (results / "settings.json").write_text(json.dumps({
        "window": window,
        "epochs": epochs,
        "runs": runs,
        "batch_size": batch_size,
        "initialization": "healthy_agent.pt",
        "training": "target stroke patient fine-tuning",
    }, indent=2))

    eeg_trials, labels, patient_ids, normalization = load_dataset(
        window, dataset_type="stroke", return_normalization=True
    )

    for index, patient in enumerate(patient_ids):
        output_dir = results / window / f"sub-{patient:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nTransfer learning: healthy -> stroke sub-{patient:02d}")
        rates = train_patient(
            eeg_trials[index], labels[index],
            epochs=epochs, runs=runs, batch_size=batch_size,
            output_dir=output_dir, normalization=normalization[index],
            seed_base=patient * 1000,
            initial_model_state=pretrained_state,
        )
        np.savez(
            output_dir / "curves.npz",
            success_rates=rates,
            mean=np.nanmean(rates, axis=0),
            std=np.nanstd(rates, axis=0, ddof=1),
        )
