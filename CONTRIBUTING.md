# 🤝 기여 가이드 (Contributing Guide)

반도체 결함 탐지 프로젝트에 기여해주셔서 감사합니다! 이 가이드는 프로젝트에 효과적으로 기여하는 방법을 설명합니다.

## 📋 목차

- [시작하기 전에](#시작하기-전에)
- [개발 환경 설정](#개발-환경-설정)
- [기여 방법](#기여-방법)
- [코딩 스타일](#코딩-스타일)
- [테스트 가이드](#테스트-가이드)
- [커밋 메시지 컨벤션](#커밋-메시지-컨벤션)
- [Pull Request 가이드](#pull-request-가이드)
- [이슈 리포팅](#이슈-리포팅)

## 🚀 시작하기 전에

### 행동 강령
우리는 모든 기여자가 존중받는 환경을 만들기 위해 노력합니다:
- 🤝 서로 존중하고 배려하기
- 💬 건설적인 피드백 제공하기
- 🎯 프로젝트 목표에 집중하기
- 📚 지속적으로 학습하고 개선하기

### 기여할 수 있는 방법
- 🐛 **버그 리포트**: 발견한 문제점 신고
- 💡 **기능 제안**: 새로운 아이디어 제안
- 📝 **문서 개선**: README, 가이드, 주석 개선
- 🧪 **모델 실험**: 새로운 아키텍처나 기법 실험
- 🔧 **코드 개선**: 성능 최적화, 리팩토링
- 🎨 **시각화**: 결과 분석 도구 개선

## 🛠️ 개발 환경 설정

### 1. 저장소 포크 및 클론
```bash
# 1. GitHub에서 저장소 Fork
# 2. 로컬에 클론
git clone https://github.com/YOUR_USERNAME/keras-.git
cd keras-

# 3. 원본 저장소를 upstream으로 추가
git remote add upstream https://github.com/LifeIsMoment/keras-.git
```

### 2. 개발 환경 구성
```bash
# 가상환경 생성
python -m venv venv311
source venv311/bin/activate  # Windows: venv311\Scripts\activate

# 개발용 의존성 설치
pip install -r requirements.txt
pip install -e ".[dev]"  # 개발용 추가 패키지

# Pre-commit 훅 설치
pre-commit install
```

### 3. 개발 도구 확인
```bash
# 코드 포맷팅 확인
black --check src tests
isort --check-only src tests

# 린팅 확인
flake8 src tests
mypy src

# 테스트 실행
pytest tests/ -v
```

## 🎯 기여 방법

### 브랜치 전략
우리는 Git Flow를 기반으로 한 브랜치 전략을 사용합니다:

```
main                    # 🏠 프로덕션 레디 코드
├── develop            # 🔧 개발 통합 브랜치
├── feature/기능명     # 🚀 새 기능 개발
├── bugfix/버그설명    # 🐛 버그 수정
├── experiment/실험명  # 🧪 모델 실험
└── hotfix/긴급수정    # 🚨 긴급 수정
```

### 워크플로우

#### 1. 이슈 확인 및 할당
```bash
# GitHub Issues에서 작업할 이슈 선택
# 이슈에 자신을 할당하거나 댓글로 작업 의사 표시
```

#### 2. 브랜치 생성
```bash
# 최신 develop 브랜치로 이동
git checkout develop
git pull upstream develop

# 기능별 브랜치 생성
git checkout -b feature/issue-number-description
# 예시: git checkout -b feature/15-resnet-transfer-learning
```

#### 3. 개발 및 커밋
```bash
# 개발 작업 수행
# 자주 커밋하기 (의미있는 단위로)
git add .
git commit -m "feat: ResNet50 전이학습 모델 구현"

# 원격 브랜치에 푸시
git push origin feature/15-resnet-transfer-learning
```

#### 4. Pull Request 생성
- GitHub에서 PR 생성
- PR 템플릿 작성
- 리뷰어 지정 (선택사항)

## 🎨 코딩 스타일

### Python 코드 스타일
우리는 [PEP 8](https://peps.python.org/pep-0008/)을 기준으로 하되, Black과 isort의 설정을 따릅니다.

#### 포맷팅 도구
```bash
# 자동 포맷팅
black src tests
isort src tests

# 검사만 (수정하지 않음)
black --check src tests
isort --check-only src tests
```

#### 네이밍 컨벤션
```python
# 클래스: PascalCase
class DefectDetectionCNN:
    pass

# 함수/변수: snake_case
def train_model():
    learning_rate = 0.001
    
# 상수: UPPER_SNAKE_CASE
MAX_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32

# 파일명: snake_case
# cnn_model.py, data_manager.py
```

#### 타입 힌트
```python
from typing import List, Dict, Optional, Tuple

def process_images(
    image_paths: List[str], 
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """이미지를 배치로 처리합니다."""
    pass

class DataManager:
    def __init__(self, config_path: str) -> None:
        self.config: Dict[str, Any] = {}
```

#### Docstring 스타일
```python
def train_model(
    model: tf.keras.Model,
    train_data: tf.data.Dataset,
    epochs: int = 100
) -> tf.keras.callbacks.History:
    """모델을 훈련합니다.

    Args:
        model: 훈련할 Keras 모델
        train_data: 훈련 데이터셋  
        epochs: 훈련 에포크 수

    Returns:
        훈련 기록을 담은 History 객체

    Raises:
        ValueError: 잘못된 입력 파라미터인 경우
        
    Example:
        >>> model = create_cnn_model()
        >>> history = train_model(model, train_ds, epochs=50)
    """
    pass
```

### 파일 구조 규칙
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/models/cnn_model.py

CNN 모델 구현을 위한 모듈입니다.
반도체 웨이퍼 결함 탐지를 위한 다양한 아키텍처를 제공합니다.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

# 로거 설정
logger = logging.getLogger(__name__)

# 클래스 구현
class DefectDetectionCNN:
    """반도체 결함 탐지를 위한 CNN 모델."""
    pass
```

## 🧪 테스트 가이드

### 테스트 구조
```
tests/
├── unit/              # 단위 테스트
│   ├── test_models.py
│   ├── test_data.py
│   └── test_utils.py
├── integration/       # 통합 테스트
│   ├── test_training.py
│   └── test_pipeline.py
└── conftest.py       # 공통 픽스처
```

### 테스트 작성 예시
```python
# tests/unit/test_models.py
import pytest
import tensorflow as tf
from src.models.cnn_model import DefectDetectionCNN

class TestDefectDetectionCNN:
    """CNN 모델 테스트 클래스."""
    
    @pytest.fixture
    def cnn_model(self):
        """테스트용 CNN 모델 픽스처."""
        return DefectDetectionCNN()
    
    def test_model_creation(self, cnn_model):
        """모델 생성 테스트."""
        model = cnn_model.build_model()
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape[1:] == (224, 224, 1)
        assert model.output_shape[1] == 9
    
    def test_model_compilation(self, cnn_model):
        """모델 컴파일 테스트.""" 
        model = cnn_model.build_model()
        assert model.optimizer is not None
        assert model.loss is not None
        
    @pytest.mark.slow
    def test_model_training(self, cnn_model, sample_data):
        """모델 훈련 테스트 (느린 테스트)."""
        # 실제 훈련 테스트
        pass
```

### 테스트 실행
```bash
# 전체 테스트
pytest tests/ -v

# 단위 테스트만
pytest tests/unit/ -v

# 빠른 테스트만 (느린 테스트 제외)
pytest tests/ -v -m "not slow"

# 커버리지 포함
pytest tests/ --cov=src --cov-report=html

# GPU 테스트 (GPU 있는 환경에서만)
pytest tests/ -v -m "gpu"
```

### 테스트 가이드라인
- **단위 테스트**: 개별 함수/클래스 테스트
- **통합 테스트**: 여러 컴포넌트 간 상호작용 테스트
- **픽스처 사용**: 공통 테스트 데이터는 conftest.py에서 관리
- **마커 활용**: `@pytest.mark.slow`, `@pytest.mark.gpu` 등으로 테스트 분류
- **모킹**: 외부 의존성은 `pytest-mock` 사용

## 📝 커밋 메시지 컨벤션

### 컨벤션 형식
```
<type>(<scope>): <subject>

<body>

<footer>
```

### 타입 (Type)
- **feat**: 새로운 기능 추가
- **fix**: 버그 수정
- **docs**: 문서 변경
- **style**: 코드 스타일 변경 (기능 변경 없음)
- **refactor**: 코드 리팩토링
- **test**: 테스트 추가 또는 수정
- **chore**: 빌드 프로세스 또는 도구 변경
- **perf**: 성능 개선
- **experiment**: 실험적 변경

### 스코프 (Scope) - 선택사항
- **data**: 데이터 처리 관련
- **model**: 모델 관련
- **training**: 훈련 관련
- **eval**: 평가 관련
- **api**: API 관련

### 예시
```bash
# 좋은 예시
feat(model): ResNet50 전이학습 모델 구현

- ImageNet 사전훈련 가중치 사용
- Fine-tuning을 위한 레이어 동결 기능 추가
- 9개 클래스 분류를 위한 출력 레이어 수정

Closes #15

fix(data): 이미지 로딩 시 메모리 누수 수정

perf(training): 배치 처리 속도 30% 개선

experiment(model): Vision Transformer 아키텍처 테스트

# 나쁜 예시
fix: bug fix
update code
added new feature
```

## 🔄 Pull Request 가이드

### PR 생성 전 체크리스트
- [ ] 최신 develop 브랜치와 동기화
- [ ] 모든 테스트 통과
- [ ] 코드 스타일 검사 통과
- [ ] 관련 문서 업데이트
- [ ] 변경사항에 대한 테스트 추가

### PR 템플릿 작성
PR 생성 시 자동으로 표시되는 템플릿을 성실히 작성해주세요:
- 변경사항 요약
- 관련 이슈 번호
- 테스트 결과
- 스크린샷 (필요시)

### 리뷰 프로세스
1. **자동 검사**: CI 파이프라인 통과 확인
2. **코드 리뷰**: 팀원의 리뷰 및 승인
3. **최종 확인**: 충돌 해결 및 최신 상태 확인
4. **머지**: develop 브랜치로 병합

### 리뷰 시 주의사항
- 🔍 **코드 품질**: 가독성, 유지보수성
- 🧪 **테스트**: 충분한 테스트 커버리지
- 📚 **문서화**: 주석 및 문서 업데이트
- 🚀 **성능**: 메모리 사용량, 속도 영향
- 🔒 **보안**: 잠재적 보안 취약점

## 🐛 이슈 리포팅

### 버그 리포트
다음 정보를 포함하여 버그를 신고해주세요:
- **환경 정보**: OS, Python 버전, GPU 정보
- **재현 단계**: 단계별 재현 방법
- **예상 동작**: 정상적으로 동작해야 하는 내용
- **실제 동작**: 실제로 발생한 문제
- **에러 메시지**: 정확한 에러 메시지

### 기능 제안
새로운 기능 제안 시 다음을 고려해주세요:
- **동기**: 왜 이 기능이 필요한가?
- **사용자 스토리**: 누가, 언제, 어떻게 사용할 것인가?
- **구현 방안**: 어떻게 구현할 수 있을까?
- **대안**: 다른 해결 방법은 없을까?

## 📊 실험 및 모델 기여

### 모델 실험 가이드
새로운 모델이나 기법을 실험할 때:

1. **실험 브랜치 생성**
```bash
git checkout -b experiment/vision-transformer-comparison
```

2. **실험 설계**
- 가설 설정
- 베이스라인 정의
- 평가 지표 선정
- 실험 조건 통제

3. **실험 실행 및 기록**
```python
# experiments/vision_transformer.py
"""Vision Transformer 실험."""

# 실험 설정
config = {
    "model": "vision_transformer",
    "patch_size": 16,
    "hidden_dim": 768,
    "num_heads": 12,
    "num_layers": 12
}

# 실험 결과 기록
results = {
    "accuracy": 0.892,
    "f1_score": 0.885,
    "inference_time": 125.3,  # ms
    "model_size": 86.5  # MB
}
```

4. **결과 문서화**
- `experiments/README.md`에 실험 내용 기록
- 성능 비교표 업데이트
- 실패한 실험도 기록 (교훈을 위해)

### 베스트 프랙티스
- 🔬 **재현 가능한 실험**: 시드 고정, 환경 명시
- 📊 **정량적 평가**: 주관적 판단보다 수치 기반
- 📝 **상세한 문서화**: 실험 과정과 결과 상세 기록
- 🔄 **반복 실험**: 신뢰성을 위한 여러 번 실행
- 🤝 **결과 공유**: 성공/실패 모두 팀과 공유

## 🎉 기여자 인정

### 기여 인정 방식
- **README.md**: 주요 기여자 목록
- **CHANGELOG.md**: 버전별 기여자 기록
- **커밋 로그**: 모든 커밋은 기여자 정보 포함
- **GitHub Contributors**: 자동 기여자 통계

### 기여 레벨
- 🥉 **Bronze**: 첫 PR 머지
- 🥈 **Silver**: 5개 이상 PR 머지
- 🥇 **Gold**: 10개 이상 PR 머지 또는 주요 기능 개발
- 💎 **Diamond**: 장기간 지속적인 기여

## 💬 소통 채널

### 질문 및 토론
- **GitHub Issues**: 버그 리포트, 기능 제안
- **GitHub Discussions**: 일반적인 질문, 아이디어 토론
- **PR Comments**: 코드 관련 구체적인 토론

### 응답 시간
- **이슈/PR**: 2-3일 내 초기 응답
- **긴급 버그**: 24시간 내 응답
- **기능 제안**: 1주일 내 검토

---

**🤝 함께 만들어가는 프로젝트** | 모든 기여에 감사드립니다!