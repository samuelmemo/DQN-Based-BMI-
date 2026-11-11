"""Train one source DQN on a combined population for transfer learning."""

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

project_code = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_code))
sys.path.insert(0, str(project_code / "stroke_DQN"))

from stroke_DQN.dataset_preparation import load_dataset
from stroke_DQN.stroke_patient_training import StrokeAgent


def train_source_agent(X, y, patient_ids, epochs=25, batch_size=8,
                       output_dir=None, normalization=None, seed=5000,
                       checkpoint_name="healthy_agent.pt",
                       initial_model_state=None):
    """Train one source agent on a combined dataset and save its checkpoint."""
    X = X.numpy()
    y = y.numpy()
    patient_ids = np.asarray(patient_ids)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Keep one full healthy pass available in replay memory.
    agent = StrokeAgent(
        chans=X.shape[2], samples=X.shape[3], batch_size=batch_size,
        buffer_capacity=len(y),
    )
    if initial_model_state is not None:
        agent.policy_net.load_state_dict(initial_model_state)
    success_rates = np.full(epochs, np.nan)
    convergence_slopes = np.full(epochs, np.nan)
    mean_changes = np.full(epochs, np.nan)
    trial_orders = np.full((epochs, len(y)), -1, dtype=np.int64)
    actions = np.full((epochs, len(y)), -1, dtype=np.int64)
    stable_windows = 0
    started = time.perf_counter()

    for epoch in range(epochs):
        # Shuffle all healthy trials for every epoch.
        trial_order = np.random.permutation(len(y))
        trial_orders[epoch] = trial_order
        successes = 0

        for trial in trial_order:
            state = X[trial]
            action = agent.select_action(state)
            actions[epoch, trial] = action
            correct = action == int(y[trial])
            reward = 1.0 if correct else -1.0
            successes += int(correct)
            agent.memory.push(state, action, reward, state, True)
            agent.update()

        success_rates[epoch] = successes / len(y)
        print(f"Epoch {epoch + 1}/{epochs}: success={success_rates[epoch]:.1%}")

        # Stop when the success-rate curve has converged, without a target value.
        if epoch >= 4:
            recent = success_rates[epoch - 4:epoch + 1]
            slope = np.polyfit(np.arange(5), recent, 1)[0]
            change = np.mean(np.abs(np.diff(recent)))
            convergence_slopes[epoch] = slope
            mean_changes[epoch] = change

            if abs(slope) <= 0.005 and change <= 0.025:
                stable_windows += 1
            else:
                stable_windows = 0

            if stable_windows >= 3:
                print(f"Converged at epoch {epoch + 1} at {recent.mean():.1%} "
                      f"(slope={slope:.4f}, mean change={change:.4f})")
                break

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": {k: v.cpu() for k, v in agent.policy_net.state_dict().items()},
            "success_rates": success_rates,
            "convergence_slopes": convergence_slopes,
            "mean_changes": mean_changes,
            "actions": actions,
            "trial_orders": trial_orders,
            "labels": y,
            "patient_ids": patient_ids,
            "normalization": normalization,
            "seed": seed,
            "epochs_completed": int(np.isfinite(success_rates).sum()),
            "stopped_early": int(np.isfinite(success_rates).sum()) < epochs,
            "elapsed_seconds": time.perf_counter() - started,
        }
        torch.save(checkpoint, output_dir / checkpoint_name)

    return success_rates


if __name__ == "__main__":
    window = "0p5s"
    results = project_code.parent / "results" / "healthy_dqn_convergence"
    eeg_trials, labels, patient_ids, normalization = load_dataset(
        window, dataset_type="healthy", return_normalization=True
    )

    # Join every healthy subject's trials into one training set.
    X = torch.cat(eeg_trials, dim=0)
    y = torch.cat(labels, dim=0)
    trial_patient_ids = np.concatenate([
        np.full(len(subject_labels), patient)
        for patient, subject_labels in zip(patient_ids, labels)
    ])

    print(f"Healthy training data: X={tuple(X.shape)}, y={tuple(y.shape)}")
    train_source_agent(
        X, y, trial_patient_ids, output_dir=results,
        normalization=normalization,
    )
