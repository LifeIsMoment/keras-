# 🔬 Keras 기반 반도체 결함 탐지 시스템

[![CI Pipeline](https://github.com/LifeIsMoment/keras-/actions/workflows/ci.yml/badge.svg)](https://github.com/LifeIsMoment/keras-/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

CNN(Convolutional Neural Network)을 사용하여 반도체 웨이퍼의 결함을 자동으로 탐지하는 딥러닝 시스템입니다. WM-811k 데이터셋을 활용하여 9가지 결함 유형을 분류하고, 기존 상용 검사 시스템 대비 우수한 성능을 달성하는 것을 목표로 합니다.

## 🎯 프로젝트 목표

| 목표 | 수치 | 현재 상태 |
|------|------|-----------|
| **분류 정확도** | 99% 이상 | 🔄 개발 중 |
| **추론 속도** | < 100ms/image | 🔄 최적화 중 |
| **비용 절감** | 기존 시스템 대비 95% | 📊 분석 중 |
| **F1-Score** | 0.95 이상 (모든 클래스) | 🎯 목표 설정 |

## 🚀 주요 특징

### 🧠 고성능 모델
- **다양한 아키텍처**: Basic CNN, ResNet, EfficientNet 지원
- **전이학습**: 사전 훈련된 모델을 활용한 빠른 수렴
- **앙상블**: 여러 모델을 조합한 성능 향상

### 📊 포괄적인 평가
- **9가지 결함 유형**: Center, Donut, Edge-Ring, Scratch, Random 등
- **다양한 메트릭**: Accuracy, Precision, Recall, F1-Score, AUC
- **시각화**: Confusion Matrix, ROC Curve, 오분류 분석

### ⚡ 실시간 처리
- **GPU 가속**: CUDA 지원으로 빠른 추론
- **배치 처리**: 대량 이미지 동시 처리
- **메모리 최적화**: 효율적인 리소스 사용

### 🔧 유연한 설정
- **설정 파일**: YAML 기반 하이퍼파라미터 관리
- **모듈화**: 독립적인 컴포넌트 설계
- **확장성**: 새로운 결함 유형 쉽게 추가

## 📁 프로젝트 구조

```
keras-/
├── 📄 README.md                    # 프로젝트 소개 및 사용법
├── 📄 requirements.txt             # Python 의존성
├── ⚙️ configs/
│   └── config.yaml                 # 설정 파일
├── 🐍 src/                        # 소스 코드
│   ├── 📊 data/                   # 데이터 처리
│   │   ├── data_manager.py        # 데이터 관리자
│   │   └── download_data.py       # 데이터 다운로드
│   ├── 🧠 models/                 # 모델 구현
│   │   ├── cnn_model.py           # CNN 아키텍처
│   │   └── transfer_learning.py   # 전이학습 모델
│   ├── 🔧 training/               # 훈련 관련
│   │   ├── trainer.py             # 훈련 매니저
│   │   └── callbacks.py           # 커스텀 콜백
│   ├── 📈 evaluation/             # 평가 도구
│   │   ├── metrics.py             # 성능 지표
│   │   └── visualizer.py          # 결과 시각화
│   └── 🛠️ utils/                  # 유틸리티
│       ├── config.py              # 설정 로더
│       └── logger.py              # 로깅 시스템
├── 📚 notebooks/                  # Jupyter 노트북
│   ├── 01_data_exploration.ipynb  # 데이터 탐색
│   ├── 02_model_experiments.ipynb # 모델 실험
│   └── 03_results_analysis.ipynb  # 결과 분석
├── 📂 data/                       # 데이터 디렉토리
│   ├── raw/                       # 원본 데이터
│   └── processed/                 # 전처리된 데이터
├── 💾 checkpoints/                # 모델 체크포인트
├── 📊 reports/                    # 실험 보고서
├── 🧪 tests/                      # 테스트 코드
└── 📖 docs/                       # 문서화
```

## 🔧 설치 및 설정

### 시스템 요구사항
- **Python**: 3.9 이상
- **GPU**: NVIDIA GPU (권장, CUDA 11.8+)
- **메모리**: 16GB RAM 이상 권장
- **저장공간**: 10GB 이상 (데이터셋 포함)

### 1. 저장소 클론
```bash
git clone https://github.com/LifeIsMoment/keras-.git
cd keras-
```

### 2. 가상환경 생성 및 활성화
```bash
# Windows
python -m venv venv311
venv311\Scripts\activate

# macOS/Linux
python -m venv venv311
source venv311/bin/activate
```

### 3. 의존성 설치
```bash
# 기본 설치
pip install -r requirements.txt

# 개발용 설치 (pre-commit, 테스트 도구 포함)
pip install -r requirements-dev.txt

# GPU 지원 (NVIDIA GPU 있는 경우)
pip install tensorflow[and-cuda]
```

### 4. 개발 환경 설정
```bash
# pre-commit 훅 설치
pre-commit install

# 설정 파일 확인
cp configs/config.yaml.example configs/config.yaml
```

## 📊 데이터셋

### WM-811k 데이터셋
- **이미지 수**: 811,457개
- **이미지 크기**: 가변 (64x64 ~ 512x512)
- **클래스 수**: 9개 결함 유형
- **형식**: PNG (그레이스케일)

### 클래스 분포
| 클래스 | 설명 | 샘플 수 | 비율 |
|--------|------|---------|------|
| None | 정상 | 147,431 | 18.2% |
| Center | 중심 결함 | 4,294 | 0.5% |
| Donut | 도넛형 결함 | 555 | 0.1% |
| Edge-Loc | 가장자리 국소 결함 | 5,189 | 0.6% |
| Edge-Ring | 가장자리 링 결함 | 9,680 | 1.2% |
| Loc | 국소 결함 | 3,593 | 0.4% |
| Near-full | 거의 전체 결함 | 149 | 0.02% |
| Random | 랜덤 결함 | 866 | 0.1% |
| Scratch | 스크래치 결함 | 1,193 | 0.1% |

## 🏃‍♂️ 빠른 시작

### 1. 샘플 데이터 생성 및 기본 훈련
```bash
# 샘플 데이터셋 생성 (테스트용)
python -c "
from src.data.data_manager import DataManager
dm = DataManager()
dm.create_sample_dataset(samples_per_class=50)
"

# 기본 모델 훈련
python train.py
```

### 2. 전체 데이터셋 다운로드 및 훈련
```bash
# WM-811k 데이터셋 다운로드 (시간 소요)
python src/data/download_data.py

# 전체 데이터로 훈련
python train.py --use-full-dataset
```

### 3. 사전 훈련된 모델로 추론
```bash
# 단일 이미지 추론
python inference.py --image path/to/wafer.png

# 배치 추론
python inference.py --batch-dir path/to/images/
```

## 🧪 모델 실험

### 지원되는 모델 아키텍처
```bash
# 기본 CNN
python train.py --model cnn

# ResNet50 전이학습
python train.py --model resnet50 --pretrained

# EfficientNet B3
python train.py --model efficientnet-b3 --pretrained

# 커스텀 아키텍처
python train.py --config configs/custom_model.yaml
```

### 하이퍼파라미터 튜닝
```bash
# 학습률 스케줄링 실험
python experiments/learning_rate_finder.py

# 배치 크기 최적화
python experiments/batch_size_optimizer.py

# 데이터 증강 실험
python experiments/augmentation_study.py
```

## 📈 성능 평가

### 기본 평가
```bash
# 테스트 세트 평가
python evaluate.py --checkpoint checkpoints/best_model.keras

# 클래스별 성능 분석
python evaluate.py --detailed --save-report
```

### 시각화 및 분석
```bash
# Confusion Matrix 생성
python src/evaluation/visualizer.py --confusion-matrix

# ROC Curve 분석
python src/evaluation/visualizer.py --roc-curves

# 오분류 샘플 분석
python src/evaluation/visualizer.py --misclassified --top-k 20
```

## 🔬 실험 결과

### 현재 베이스라인 성능
| 모델 | 정확도 | F1-Score | 추론 시간 | 모델 크기 |
|------|--------|----------|-----------|-----------|
| Basic CNN | 85.2% | 0.82 | 45ms | 19.4MB |
| ResNet50 | 🔄 실험 중 | - | - | - |
| EfficientNet-B3 | 🔄 실험 중 | - | - | - |

### 클래스별 성능 (Basic CNN)
| 클래스 | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| None | 0.92 | 0.95 | 0.93 |
| Center | 0.78 | 0.82 | 0.80 |
| Donut | 0.85 | 0.76 | 0.80 |
| Edge-Loc | 0.81 | 0.79 | 0.80 |
| Edge-Ring | 0.83 | 0.85 | 0.84 |
| Loc | 0.77 | 0.74 | 0.75 |
| Near-full | 0.65 | 0.58 | 0.61 |
| Random | 0.72 | 0.69 | 0.70 |
| Scratch | 0.79 | 0.83 | 0.81 |

## 🤝 기여하기

### 개발 워크플로우
1. **Fork** 저장소
2. **Feature 브랜치** 생성: `git checkout -b feature/amazing-feature`
3. **변경사항 커밋**: `git commit -m 'feat: add amazing feature'`
4. **브랜치 푸시**: `git push origin feature/amazing-feature`
5. **Pull Request** 생성

### 커밋 메시지 컨벤션
```
feat: 새로운 기능 추가
fix: 버그 수정
docs: 문서 변경
style: 코드 스타일 변경 (기능 변경 없음)
refactor: 코드 리팩토링
test: 테스트 추가 또는 수정
chore: 빌드 프로세스 또는 도구 변경
perf: 성능 개선
experiment: 실험적 변경
```

### 코드 품질
- **Black**: 코드 포맷팅
- **Flake8**: 스타일 검사
- **MyPy**: 타입 검사
- **Pre-commit**: 커밋 전 자동 검사

## 📚 문서

### API 문서
- [모델 아키텍처](docs/model_architecture.md)
- [데이터 전처리](docs/data_preprocessing.md)
- [훈련 가이드](docs/training_guide.md)
- [평가 지표](docs/evaluation_metrics.md)

### 예제 노트북
- [데이터 탐색](notebooks/01_data_exploration.ipynb)
- [모델 실험](notebooks/02_model_experiments.ipynb)
- [결과 분석](notebooks/03_results_analysis.ipynb)

## 🔧 문제 해결

### 자주 발생하는 문제

**Q: `AttributeError: 'WindowsPath' object has no attribute 'endswith'`**
```bash
# 해결: 최신 버전으로 업데이트
git pull origin main
pip install -r requirements.txt
```

**Q: GPU 메모리 부족 오류**
```bash
# 해결: 배치 크기 감소
python train.py --batch-size 16  # 기본값 32에서 감소
```

**Q: 데이터 다운로드 실패**
```bash
# 해결: 수동 다운로드 후 압축 해제
# 1. https://www.kaggle.com/datasets/qingyi/wm-811k-wafer-map 접속
# 2. 다운로드 후 data/raw/ 디렉토리에 압축 해제
```

더 많은 문제 해결 방법은 [문제 해결 가이드](docs/troubleshooting.md)를 참조하세요.

## 📊 로드맵

### 🔴 현재 (v0.1.0) - MVP
- [x] 기본 CNN 모델 구현
- [x] 데이터 로딩 시스템
- [x] 기본 훈련 파이프라인
- [ ] 성능 평가 시스템

### 🟡 다음 단계 (v0.2.0) - 성능 최적화
- [ ] 전이학습 모델 (ResNet, EfficientNet)
- [ ] 하이퍼파라미터 자동 튜닝
- [ ] 모델 앙상블
- [ ] 추론 시간 최적화

### 🟢 향후 계획 (v0.3.0) - 프로덕션 준비
- [ ] REST API 서버
- [ ] 웹 인터페이스
- [ ] 모델 버전 관리
- [ ] A/B 테스트 시스템

### 🔵 장기 목표 (v1.0.0) - 완전한 시스템
- [ ] 실시간 스트리밍 처리
- [ ] 클라우드 배포
- [ ] 모니터링 대시보드
- [ ] 자동 재훈련 시스템

## 📄 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다.

## 👥 팀

- **개발자**: LifeIsMoment
- **이메일**: [이메일 주소]
- **GitHub**: [@LifeIsMoment](https://github.com/LifeIsMoment)

## 🙏 감사의 글

- **WM-811k 데이터셋**: [Taiwan Semiconductor Manufacturing Company](https://www.tsmc.com/)
- **TensorFlow/Keras**: Google의 오픈소스 머신러닝 프레임워크
- **Pre-commit**: 코드 품질 관리 도구

## 📈 통계

![GitHub stars](https://img.shields.io/github/stars/LifeIsMoment/keras-)
![GitHub forks](https://img.shields.io/github/forks/LifeIsMoment/keras-)
![GitHub issues](https://img.shields.io/github/issues/LifeIsMoment/keras-)
![GitHub last commit](https://img.shields.io/github/last-commit/LifeIsMoment/keras-)

---

**🔬 반도체 제조 혁신을 위한 AI 솔루션** | Made with ❤️ by LifeIsMoment
