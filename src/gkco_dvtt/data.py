from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class DatasetBundle:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray

    @property
    def all_x(self) -> np.ndarray:
        return np.concatenate([self.train_x, self.val_x, self.test_x], axis=0)

    @property
    def all_y(self) -> np.ndarray:
        return np.concatenate([self.train_y, self.val_y, self.test_y], axis=0)

    @property
    def split_sizes(self) -> tuple[int, int, int]:
        return len(self.train_x), len(self.val_x), len(self.test_x)


def _load_array(root: Path, name: str) -> np.ndarray:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return np.load(path)


def load_numpy_dataset(root: str | Path) -> DatasetBundle:
    root = Path(root)
    bundle = DatasetBundle(
        train_x=_load_array(root, "train_x.npy"),
        train_y=_load_array(root, "train_y.npy"),
        val_x=_load_array(root, "validation_x.npy"),
        val_y=_load_array(root, "validation_y.npy"),
        test_x=_load_array(root, "test_x.npy"),
        test_y=_load_array(root, "test_y.npy"),
    )
    _validate_bundle(bundle)
    return bundle


def _validate_bundle(bundle: DatasetBundle) -> None:
    pairs = [
        (bundle.train_x, bundle.train_y, "train"),
        (bundle.val_x, bundle.val_y, "validation"),
        (bundle.test_x, bundle.test_y, "test"),
    ]
    feature_dim = None
    for x, y, split in pairs:
        if x.ndim != 2:
            raise ValueError(f"{split}_x must have shape [N, L], got {x.shape}")
        if y.ndim != 1:
            y = np.asarray(y).reshape(-1)
        if len(x) != len(y):
            raise ValueError(f"{split} features and labels have different lengths")
        if feature_dim is None:
            feature_dim = x.shape[1]
        elif x.shape[1] != feature_dim:
            raise ValueError("All splits must use the same sample length")


def build_masks(split_sizes: tuple[int, int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_train, n_val, n_test = split_sizes
    n_total = n_train + n_val + n_test
    train_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    return train_mask, val_mask, test_mask
