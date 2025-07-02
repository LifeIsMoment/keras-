#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/models/cnn_model.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
CNN architecture + callbacks for wafer‑defect classification.

Changes vs. previous version
----------------------------
* `ModelCheckpoint` now receives a *string* file path (converted from Path object) 
  to avoid `AttributeError: 'WindowsPath' object has no attribute 'endswith'`.
* Using `.keras` extension as recommended by Keras 3.
"""
from __future__ import annotations

import yaml
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from tensorflow.keras.metrics import Precision, Recall


class DefectDetectionCNN:
    """Lightweight CNN for 224×224 grayscale wafer maps."""

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.input_shape: tuple[int, int, int] = tuple(self.config["model"]["input_shape"])
        self.num_classes: int = self.config["model"]["num_classes"]
        self.learning_rate: float = self.config["model"]["learning_rate"]

        self.model: tf.keras.Model | None = None

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------

    def build_model(self) -> tf.keras.Model:
        """Create and compile the CNN architecture."""
        model = Sequential(
            [
                # Block 1
                Conv2D(32, (3, 3), activation="relu", input_shape=self.input_shape),
                BatchNormalization(),
                MaxPooling2D(2),
                # Block 2
                Conv2D(64, 3, activation="relu"),
                BatchNormalization(),
                MaxPooling2D(2),
                # Block 3
                Conv2D(128, 3, activation="relu"),
                BatchNormalization(),
                MaxPooling2D(2),
                # Block 4
                Conv2D(256, 3, activation="relu"),
                BatchNormalization(),
                MaxPooling2D(2),
                # Classification head
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
            metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
        )

        self.model = model
        return model

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def summary(self) -> None:  # convenience alias
        if self.model is None:
            self.build_model()
        self.model.summary()

    def prepare_callbacks(self):
        """Return a typical EarlyStopping / ReduceLROnPlateau / Checkpoint combo."""
        checkpoint_dir = Path(self.config["training"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        return [
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
                filepath=str(checkpoint_dir / "best_model.keras"),  # Path를 문자열로 변환
                save_best_only=self.config["training"]["save_best_only"],
                verbose=1,
            ),
        ]


if __name__ == "__main__":
    print("🏗️  CNN 모델 생성 테스트")
    cnn = DefectDetectionCNN()
    cnn.build_model()
    print("\n📋 모델 구조:")
    cnn.summary()
    print("\n✅ CNN 모델 생성 완료!")