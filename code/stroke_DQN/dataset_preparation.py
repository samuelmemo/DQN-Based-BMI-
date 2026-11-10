
from pathlib import Path

import numpy as np
import torch
import h5py


BMI_input_path = (
    Path(__file__).resolve().parents[2]
    / "data" / "Results" / "BMI_inputs" / "stroke"
)


def load_dataset(window, return_normalization=False):
    eeg_trials, labels, patient_ids = [], [], []
    normalization = []

    for patient in range(1, 51):
        path = BMI_input_path / f"sub-{patient:02d}_BMI_input_{window}.mat"

        # h5py reads MATLAB v7.3 dimensions as (channels, time, trials).
        with h5py.File(path, "r") as data:
            X = np.asarray(data["InputTensor"], dtype=np.float32)
            y = np.asarray(data["labels"], dtype=np.int64).reshape(-1) - 1

        # Convert to PyTorch: (trials, 1, channels, time)
        X = torch.from_numpy(X.transpose(2, 0, 1).copy()).unsqueeze(1)
        y = torch.from_numpy(y)

        if X.shape[0] != len(y):
            raise ValueError(f"Trial/label mismatch in {path.name}")

        # One mean/std per channel, across this patient's trials and time.
        mean = X.mean(dim=(0, 3), keepdim=True)
        std = X.std(dim=(0, 3), keepdim=True, unbiased=False)

        X = (X - mean) / (std + 1e-8)
        normalization.append({"mean": mean, "std": std, "eps": 1e-8})

        eeg_trials.append(X)
        labels.append(y)
        patient_ids.append(patient)

    if return_normalization:
        return eeg_trials, labels, patient_ids, normalization
    return eeg_trials, labels, patient_ids


if __name__ == "__main__":
    data_1s = load_dataset("1p0s")
    data_05s = load_dataset("0p5s")

    X, y = data_1s[0][0], data_1s[1][0]

    print("EEG shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Labels:", torch.unique(y))

    
