"""Backward-compatible wrapper for the shared dataset loader."""

from dataset_preparation import load_dataset


def load_dataset_healthy(window, return_normalization=False):
    return load_dataset(
        window,
        dataset_type="healthy",
        return_normalization=return_normalization,
    )
