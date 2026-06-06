from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd


class NYU2KaggleDataset:
    """
    Loader for Kaggle version of NYU Depth Dataset V2.

    Expected project structure:

    rgbd-free-space-mapping/
    ├── data/
    │   └── raw/
    │       ├── nyu2_train/
    │       ├── nyu2_test/
    │       ├── nyu2_train.csv
    │       └── nyu2_test.csv

    CSV format:
        data/nyu2_train/living_room_0038_out/37.jpg,data/nyu2_train/living_room_0038_out/37.png
        data/nyu2_test/00000_colors.png,data/nyu2_test/00000_depth.png
    """

    def __init__(
        self,
        data_root: str | Path = "data/raw",
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split.lower()

        if self.split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'")

        self.csv_path = self.data_root / f"nyu2_{self.split}.csv"

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_path}. " f"Expected file: data/raw/nyu2_{self.split}.csv"
            )

        self.samples = self._read_csv(self.csv_path)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.csv_path}")

    def _read_csv(self, csv_path: Path) -> List[Tuple[Path, Path]]:
        """
        Reads CSV file without header.
        Each row contains RGB image path and depth image path.
        """
        df = pd.read_csv(csv_path, header=None)

        if df.shape[1] < 2:
            raise ValueError(
                f"CSV file must contain at least 2 columns: rgb_path, depth_path. " f"Got shape: {df.shape}"
            )

        samples: List[Tuple[Path, Path]] = []

        for _, row in df.iterrows():
            rgb_path_raw = str(row.iloc[0])
            depth_path_raw = str(row.iloc[1])

            rgb_path = self._resolve_dataset_path(rgb_path_raw)
            depth_path = self._resolve_dataset_path(depth_path_raw)

            samples.append((rgb_path, depth_path))

        return samples

    def _resolve_dataset_path(self, path_from_csv: str) -> Path:
        """
        Resolves paths from CSV.

        Kaggle CSV often stores paths like:
            data/nyu2_train/...
            data/nyu2_test/...

        In this project we store dataset inside:
            data/raw/nyu2_train/...
            data/raw/nyu2_test/...

        Therefore, this function removes leading 'data/' if needed
        and appends the remaining path to data_root.
        """
        normalized = path_from_csv.replace("\\", "/")

        if normalized.startswith("data/"):
            normalized = normalized[len("data/") :]

        return self.data_root / normalized

    def __len__(self) -> int:
        return len(self.samples)

    def get_paths(self, index: int) -> Tuple[Path, Path]:
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.samples)}")

        return self.samples[index]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        rgb_path, depth_path = self.get_paths(index)

        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")

        if not depth_path.exists():
            raise FileNotFoundError(f"Depth image not found: {depth_path}")

        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"Failed to read RGB image: {rgb_path}")

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Failed to read depth image: {depth_path}")

        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

        depth = depth.astype(np.float32)

        return rgb, depth

    def get_sample_info(self, index: int) -> dict:
        rgb_path, depth_path = self.get_paths(index)
        rgb, depth = self[index]

        return {
            "index": index,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "rgb_shape": rgb.shape,
            "depth_shape": depth.shape,
            "depth_dtype": str(depth.dtype),
            "depth_min": float(np.nanmin(depth)),
            "depth_max": float(np.nanmax(depth)),
            "depth_mean": float(np.nanmean(depth)),
        }
