# Keras 기반 반도체 부품 검사 시스템

Keras를 활용한 딥러닝 기반 반도체 결함 탐지 시스템 개발 프로젝트

## 프로젝트 개요

본 프로젝트는 CNN(Convolutional Neural Network)을 사용하여 반도체 웨이퍼의 결함을 자동으로 탐지하는 시스템을 개발합니다. WM-811k 데이터셋을 활용하여 9가지 결함 유형을 분류하고, 기존 상용 검사 시스템 대비 우수한 성능을 입증하는 것을 목표로 합니다.

## 주요 목표

- **높은 정확도**: 99% 이상의 결함 탐지 정확도 달성
- **실시간 처리**: 빠른 추론 속도로 실시간 검사 지원
- **비용 효율성**: 기존 하드웨어 기반 시스템 대비 95% 비용 절감
- **유연성**: 다양한 결함 유형에 대한 커스터마이징 가능

## 프로젝트 구조
keras-/

├── README.md

├── requirements.txt

├── configs/

│   └── config.yaml

├── src/

│   ├── data/              # 데이터 처리

│   ├── models/            # 모델 구현

│   ├── features/          # 특성 엔지니어링

│   ├── visualization/     # 시각화

│   └── utils/            # 유틸리티

├── notebooks/            # Jupyter 노트북

├── data/                # 데이터셋

├── reports/             # 보고서 및 결과

├── tests/               # 테스트 코드

├── docs/                # 문서

└── checkpoints/         # 모델 저장소

## 빠른 시작

1. 저장소 클론
```bash
git clone https://github.com/LifeIsMoment/keras-.git
cd keras-
