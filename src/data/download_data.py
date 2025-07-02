#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/models/cnn_model.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
CNN architecture & training callbacks for wafer‑defect classification.

Key fix 2025‑07‑02
------------------
* **ModelCheckpoint** now receives a *string* file path that ends with
  ".keras" (required by Keras 3) to avoid
  `AttributeError: 'WindowsPath' object has no attribute 'endswith'`.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class DefectDetectionCNN:
    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.input_shape = tuple(self.config["model"]["input_shape"])
        self.num_classes = self.config["model"]["num_classes"]
        self.learning_rate = self.config["model"]["learning_rate"]

        self.model = self._build_model()

    # ------------------------------------------------------------------
    # 1. Model definition
    # ------------------------------------------------------------------
    def _build_model(self):
        model = Sequential(
            [
                Conv2D(32, (3, 3), activation="relu", input_shape=self.input_shape),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Conv2D(64, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Conv2D(128, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Conv2D(256, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(256, activation="relu"),
                Dropout(0.3),
                Dense(self.num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",  # shorthand OK
                Precision(name="precision"),
                Recall(name="recall"),
            ],
        )
        return model

    # ------------------------------------------------------------------
    # 2. Public helpers
    # ------------------------------------------------------------------
    def summary(self):
        return self.model.summary()

    def prepare_callbacks(self):
        ckpt_dir = Path(self.config["training"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EarlyStopping(
                patience=self.config["model"]["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                factor=0.2,
                patience=self.config["model"].get("reduce_lr_patience", 10),
                min_lr=1e-7,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(ckpt_dir / "best_model.keras"),  # ← FIX: str + .keras
                save_best_only=self.config["training"]["save_best_only"],
                verbose=1,
            ),
        ]
        return callbacks


# ----------------------------------------------------------------------
# Quick standalone test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("🏗️  CNN 모델 생성 테스트")
    cnn = DefectDetectionCNN()
    cnn.summary()
    print("✅ CNN 모델 생성 완료!")
