"""Train a fresh EEGNet DQN on one stroke patient, without transfer learning."""

import sys
import csv
import json
import random
import time
import shutil
from pathlib import Path

import numpy as np
import torch

from dataset_preparation import load_dataset

# Reuse the original network architecture and replay buffer.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "freewill-eeg-qlearning"))
from models import EEGNet, ReplayBuffer


class StrokeAgent:
    """One-step Q-learning: the target is just the observed reward."""

    def __init__(self, chans, samples, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = EEGNet(2, Chans=chans, Samples=samples, kernLength=80).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayBuffer(1000)
        self.batch_size = batch_size

    def select_action(self, state):
        # Explore randomly on 1% of decisions.
        if np.random.rand() < 0.01:
            return int(np.random.randint(2))
        self.policy_net.eval()  #Put EEGNet into evaluation mode
        with torch.no_grad():   #Don't track gradients
            state = torch.as_tensor(state, device=self.device).unsqueeze(0)
            return self.policy_net(state).argmax().item()   #Return estimated Q-values 

    def update(self):
        if len(self.memory) < self.batch_size:   #Wait until it has enough stored experiences
            return
        states, actions, rewards, _, _ = self.memory.sample(self.batch_size)  #Sample experiences from replay buffer
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Train only the chosen action's Q-value against its immediate reward.
        self.policy_net.train()  #Put EEGNet into training mode
        predictions = self.policy_net(states).gather(1, actions)  #Extract only the Q-value corresponding to the selected action
        loss = torch.nn.functional.mse_loss(predictions, rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train_patient(X, y, epochs=25, runs=10, batch_size=8,
                  output_dir=None, normalization=None, seed_base=1000):
    # The original replay buffer expects NumPy arrays, not PyTorch tensors.
    X = X.numpy()
    y = y.numpy()
    # Uncompleted epochs stay missing, rather than looking like failures.
    success_rates = np.full((runs, epochs), np.nan)

    for run in range(runs):
        # Resume at completed-run boundaries, never in the middle of a run.
        run_path = output_dir / f"run-{run + 1:02d}.pt" if output_dir else None
        if run_path and run_path.exists():
            saved = torch.load(run_path, map_location="cpu", weights_only=False)
            success_rates[run] = saved["success_rates"]
            print(f"Skipping completed {run_path}")
            continue
        seed = seed_base + run
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        started = time.perf_counter()
        # Start each run with a new network and an empty replay buffer.
        agent = StrokeAgent(chans=X.shape[2], samples=X.shape[3], batch_size=batch_size)
        # Save each epoch's order; -1 marks epochs that were not run.
        trial_orders = np.full((epochs, len(y)), -1, dtype=np.int64)
        actions = np.full((epochs, len(y)), -1, dtype=np.int64)

        for epoch in range(epochs):
            # Present the patient's trials in a fresh random order each epoch.
            trial_order = np.random.permutation(len(y))
            trial_orders[epoch] = trial_order
            successes = 0
            for trial in trial_order:
                state = X[trial]
                action = agent.select_action(state)
                actions[epoch, trial] = action

                # With two opposite targets, a correct action is a successful reach.
                correct = action == int(y[trial])
                reward = 1.0 if correct else -1.0
                successes += int(correct)

                # Each EEG trial is a terminal, one-decision episode.
                agent.memory.push(state, action, reward, state, True)
                agent.update()

            success_rates[run, epoch] = successes / len(y)
            print(f"Run {run + 1}/{runs}, epoch {epoch + 1}/{epochs}: "
                  f"success={success_rates[run, epoch]:.1%}")
            # Stop once four of the last five epochs reach at least 95% success.
            if epoch >= 4 and np.count_nonzero(success_rates[run, epoch - 4:epoch + 1] >= 0.95) >= 4:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if run_path:
            # Save atomically: an interrupted write is not mistaken for a finished run.
            checkpoint = {
                "model_state": {k: v.cpu() for k, v in agent.policy_net.state_dict().items()},
                "success_rates": success_rates[run], "actions": actions,
                "labels": y, "trial_orders": trial_orders, "seed": seed,
                "normalization": normalization,
                "epochs_completed": epoch + 1,
                "stopped_early": epoch + 1 < epochs,
                "elapsed_seconds": time.perf_counter() - started,
            }
            temporary = run_path.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(run_path)

    # Each column contains the results of the independent runs for one epoch.
    return success_rates


if __name__ == "__main__":
    # Use a new experiment folder if you change settings or input files.
    results = Path(__file__).resolve().parents[2] / "results" / "stroke_dqn_earlystop95_buffer1000_shuffle"
    results.mkdir(parents=True, exist_ok=True)
    settings = {
        "epochs": 25, "runs": 10, "batch_size": 8, "kernel_samples": 80,
        "replay_capacity": 1000,
        "learning_rate": 0.001, "epsilon": 0.01, "rewards": [1, -1],
        "sampling_rate": 160, "normalization": "per patient/channel over trials and time",
        "metric": "online training success, exploration included",
        "early_stopping": "4 of last 5 epochs at or above 95% success; maximum 25 epochs",
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "seed_rule": "patient * 1000 + zero-based run; same seeds for both windows",
        "trial_order": "reshuffled each epoch",
    }
    config_path = results / "settings.json"
    if config_path.exists() and json.loads(config_path.read_text()) != settings:
        raise ValueError("Saved settings differ. Choose a new results folder.")
    config_path.write_text(json.dumps(settings, indent=2))
    # Keep copies of the code that produced the results.
    for source in (Path(__file__), Path(__file__).with_name("dataset_preparation.py"),
                   Path(__file__).resolve().parents[1] / "freewill-eeg-qlearning" / "models.py"):
        destination = results / source.name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError("Saved code differs. Choose a new results folder.")
        shutil.copyfile(source, destination)

    for window in ("1p0s", "0p5s"):
        eeg_trials, labels, patient_ids, normalization = load_dataset(window, return_normalization=True)
        window_dir = results / window
        window_dir.mkdir(exist_ok=True)
        summary = []
        for index, patient in enumerate(patient_ids):
            output_dir = window_dir / f"sub-{patient:02d}"
            output_dir.mkdir(exist_ok=True)
            print(f"\nWindow {window}, patient {patient:02d}")
            rates = train_patient(
                eeg_trials[index], labels[index], epochs=settings["epochs"],
                runs=settings["runs"], batch_size=settings["batch_size"],
                output_dir=output_dir, normalization=normalization[index], seed_base=patient * 1000,
            )
            # Raw run curves plus mean/std for easy plotting (rates are fractions).
            completed = np.isfinite(rates).sum(axis=1)
            # Mask missing epochs so summaries use only observed results.
            observed = np.ma.masked_invalid(rates)
            np.savez(output_dir / "curves.npz", success_rates=rates,
                     epochs_completed=completed,
                     mean=observed.mean(axis=0).filled(np.nan),
                     std=observed.std(axis=0, ddof=1).filled(np.nan))
            final = np.array([row[n - 1] for row, n in zip(rates, completed)])
            summary.append([patient, rates[:, 0].mean(), rates[:, 0].std(ddof=1),
                            final.mean(), final.std(ddof=1), completed.mean()])
            with (window_dir / "summary.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["patient", "first_epoch_mean", "first_epoch_std",
                                 "final_epoch_mean", "final_epoch_std", "mean_epochs_completed"])
                writer.writerows(summary)
