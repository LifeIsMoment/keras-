#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 훈련 실행 스크립트
~~~~~~~~~~~~~~~~~~~~~~
Run from the project root:
    python train.py

Requirements
------------
* `src/` must be on PYTHONPATH so that `data` and `models` packages are
  import‑able.
* `configs/config.yaml` must exist and be **UTF‑8 encoded**.
"""
import pathlib
import sys

import yaml
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# 1.  project paths & dynamic PYTHONPATH injection
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent  # <project_root>
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:  # 중복 방지
    sys.path.insert(0, str(SRC_DIR))

from data.data_manager import DataManager  # alias of SemiconductorDataManager

# 이제 내부 패키지 import 가 안전하게 동작한다
from models.cnn_model import DefectDetectionCNN

# ---------------------------------------------------------------------------
# 2.  main routine
# ---------------------------------------------------------------------------


def main() -> None:
    print("🚀 모델 훈련 시작!")

    # -------------------------------------------------------------------
    # 2‑1. Load config (always UTF‑8!)
    # -------------------------------------------------------------------
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # -------------------------------------------------------------------
    # 2‑2. Verify / create sample dataset
    # -------------------------------------------------------------------
    data_manager = DataManager()
    print("📁 데이터 확인 중…")

    # -------------------------------------------------------------------
    # 2‑3. Build model
    # -------------------------------------------------------------------
    cnn = DefectDetectionCNN()
    model = cnn.build_model()

    print("📋 모델 구조:")
    model.summary()

    # -------------------------------------------------------------------
    # 2‑4. Data generators
    # -------------------------------------------------------------------
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )

    train_generator = datagen.flow_from_directory(
        "data/raw/train",
        target_size=(224, 224),
        batch_size=config["data"]["batch_size"],
        class_mode="categorical",
        color_mode="grayscale",
    )

    val_generator = datagen.flow_from_directory(
        "data/raw/validation",
        target_size=(224, 224),
        batch_size=config["data"]["batch_size"],
        class_mode="categorical",
        color_mode="grayscale",
    )

    print(f"📊 훈련 데이터: {train_generator.samples} 샘플")
    print(f"📊 검증 데이터: {val_generator.samples} 샘플")
    print(f"📊 클래스: {list(train_generator.class_indices.keys())}")

    # -------------------------------------------------------------------
    # 2‑5. Callbacks & training
    # -------------------------------------------------------------------
    callbacks = cnn.prepare_callbacks()

    print("🏃‍♂️ 훈련 시작…")
    history = model.fit(
        train_generator,
        epochs=5,  # quick smoke‑test; increase for real training
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1,
    )

    print("✅ 훈련 완료!")

    # -------------------------------------------------------------------
    # 2‑6. Save model
    # -------------------------------------------------------------------
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    model.save(str(ROOT / "checkpoints" / "trained_model.keras"))
    print("💾 모델 저장 완료 → checkpoints/trained_model.h5")


if __name__ == "__main__":
    main()
