# MILE: Model-based Intervention Learning Framework

이 프로젝트는 **Model-based Intervention Learning (MILE)** 알고리즘을 다양한 강화학습 환경(LunarLander, CarRacing 등)에서 학습해보기 위해 구축된 프레임워크입니다. 멘탈 모델 기반의 개입 학습 파이프라인을 직접 전체 로직으로 구현하였습니다.

## 주요 특징
* **Modular Pipeline:** `RL 사전학습` -> `Behavioral Cloning(멘탈 모델 초기화)` -> `MILE 라운드 학습`의 파이프라인을 독립적으로 관리.
* **Compatibility:** 이산형(Discrete) 환경과 연속형(Continuous) 환경을 `config.json`을 통해 손쉽게 전환.
* **Optimization:** PyTorch 기반의 커스텀 손실 함수와 최적화 로직 적용.
* **Efficiency:** 대용량 이미지 관측값을 위한 메모리 최적화 옵션 및 다중 디바이스(CUDA/MPS/CPU) 자동 할당 지원.

> **Note:** 본 프레임워크의 핵심 로직과 파이프라인 설계는 직접 구현하였으며, 코드의 세부 구현 과정에서 AI의 기술적 도움(디버깅 및 최적화)을 일부 참고하였습니다.

## 학습 방법 (예시: CarRacing-v2 env)
1. **의존성 설치:**
```bash
   pip install -r requirements.txt
```
2. **Config 설정:** `config.json`에서 환경별 하이퍼파라미터 및 `net_arch`를 설정합니다.
3. **사전 학습:** 
```bash
   python scripts/train_initial.py --env_type continuous --frame_stack 4
```
4. **멘탈 모델 초기화(BC):** 
```bash
   python scripts/create_bc.py --env_type continuous --frame_stack 4
```
5. **MILE 학습:** 
```bash
   python scripts/train_mile.py --env_type continuous --frame_stack 4
```
6. **테스트:** 
```bash
   python scripts/test_model.py --env_type continuous --frame_stack 4 --round 5
```

## 환경 정보
* **LunarLander-v2 (Discrete, MLP, DQN)**

* **CarRacing-v2 (Continuous, CNN-based, SAC)**
