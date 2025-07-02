#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/data/data_manager.py
================================================
Synthetic wafer‑defect dataset utilities.

▪ SemiconductorDataManager  – core class used across the project
▪ DataManager               – convenient alias so you can simply write
                                `from data.data_manager import DataManager`

Usage (from project root):
    >>> from data.data_manager import DataManager
    >>> dm = DataManager()
    >>> dm.create_sample_dataset(samples_per_class=100)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

__all__ = ["SemiconductorDataManager", "DataManager"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class SemiconductorDataManager:
    """Manage raw / synthetic wafer‑defect datasets."""

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        with open(config_path, encoding="utf-8") as f:
            self.config: Dict = yaml.safe_load(f)

        self.raw_data_path: Path = Path(self.config["data"]["raw_data_path"])
        self.processed_data_path: Path = Path(
            self.config["data"]["processed_data_path"]
        )
        self.classes: List[str] = self.config["classes"]

        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        logger.info("데이터 매니저 초기화 완료 → %s", self.raw_data_path)

    # ------------------------------------------------------------------
    # Dataset creation helpers
    # ------------------------------------------------------------------
    def create_sample_dataset(self, samples_per_class: int = 100) -> None:
        """Generate a small synthetic dataset for quick experiments."""
        logger.info("샘플 데이터셋 생성 (%d×%d 클래스)", samples_per_class, len(self.classes))
        splits = ["train", "validation", "test"]
        ratios = [0.7, 0.2, 0.1]

        for split, ratio in zip(splits, ratios):
            n_samples = int(samples_per_class * ratio)
            split_dir = self.raw_data_path / split
            split_dir.mkdir(exist_ok=True)

            for cls in self.classes:
                cls_dir = split_dir / cls
                cls_dir.mkdir(exist_ok=True)
                for i in tqdm(range(n_samples), desc=f"{split}/{cls}"):
                    img = self._generate_synthetic_wafer(cls)
                    Image.fromarray(img, mode="L").save(
                        cls_dir / f"{cls.lower()}_{i:04d}.png"
                    )

        self._save_dataset_info(samples_per_class)
        logger.info("Synthetic dataset ready! → %s", self.raw_data_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_synthetic_wafer(
        self, defect_type: str, size: tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """Return a single‑channel (uint8) synthetic wafer image for *defect_type*."""
        img = np.clip(np.random.normal(128, 20, size), 50, 200).astype(np.uint8)
        cx, cy = size[0] // 2, size[1] // 2
        yy, xx = np.ogrid[: size[0], : size[1]]

        if defect_type == "Center":
            r = np.random.randint(15, 25)
            img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r**2] = np.random.randint(20, 40)
        elif defect_type == "Donut":
            ro, ri = np.random.randint(30, 45), np.random.randint(15, 25)
            outer = (xx - cx) ** 2 + (yy - cy) ** 2 <= ro**2
            inner = (xx - cx) ** 2 + (yy - cy) ** 2 <= ri**2
            img[outer & ~inner] = np.random.randint(20, 40)
        elif defect_type == "Scratch":
            y0, y1 = np.random.randint(10, size[0] // 3), np.random.randint(
                2 * size[0] // 3, size[0] - 10
            )
            x0, t = np.random.randint(
                size[1] // 4, 3 * size[1] // 4
            ), np.random.randint(2, 5)
            img[y0:y1, x0 - t : x0 + t] = np.random.randint(15, 35)
        elif defect_type == "Edge-Ring":
            edge = np.minimum.reduce([xx, yy, size[1] - 1 - xx, size[0] - 1 - yy])
            w = np.random.randint(8, 15)
            img[(edge >= 5) & (edge <= 5 + w)] = np.random.randint(20, 40)
        elif defect_type == "Random":
            for _ in range(np.random.randint(5, 15)):
                x, y, r = (
                    np.random.randint(10, size[1] - 10),
                    np.random.randint(10, size[0] - 10),
                    np.random.randint(2, 6),
                )
                mask = (xx - x) ** 2 + (yy - y) ** 2 <= r**2
                img[mask] = np.random.randint(15, 35)
        return img

    def _save_dataset_info(self, samples_per_class: int) -> None:
        info = {
            "dataset_name": "Synthetic Semiconductor Defect Dataset",
            "classes": self.classes,
            "num_classes": len(self.classes),
            "samples_per_class": samples_per_class,
            "image_size": self.config["data"]["image_size"],
        }
        out = self.raw_data_path / "dataset_info.yaml"
        with open(out, "w", encoding="utf-8") as f:
            yaml.dump(info, f, allow_unicode=True)
        logger.info("메타데이터 저장 → %s", out)


# ----------------------------------------------------------------------
# Public alias for compatibility with earlier scripts
# ----------------------------------------------------------------------
DataManager = SemiconductorDataManager
