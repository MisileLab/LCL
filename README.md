# LLM용 zram: Long-context LLM을 위한 (Near-)Lossless KV 캐시 압축 레이어

SmolLM3-3B-128k 기반 KV 캐시 압축 시스템 구현

## 프로젝트 개요

본 프로젝트는 Long-context LLM의 KV 캐시를 효율적으로 압축하여 메모리 사용량을 줄이면서도 성능 저하를 최소화하는 시스템입니다.

### 주요 특징

- **Level 1 압축 (GPU-native)**: Transform + Near-lossless Quantization + Bit Packing
- **Level 2 압축 (CPU/오프라인)**: 페이지 단위 엔트로피 코딩
- **SmolLM3-3B-128k** 타겟 모델
- **컨텍스트 길이**: 8k / 32k / 128k 지원

## 아키텍처

```
┌─────────────────────────────────────────┐
│         Inference Engine                │
│  (Prefill + Decode with Custom KV)      │
└──────────────┬──────────────────────────┘
               │
       ┌───────▼────────┐
       │  Page Manager  │
       │  (Hot/Cold)    │
       └───┬────────┬───┘
           │        │
    ┌──────▼──┐  ┌─▼─────────┐
    │ Level 1 │  │  Level 2  │
    │  (GPU)  │  │  (CPU)    │
    └─────────┘  └───────────┘
```

## 설치

```bash
pip install -r requirements.txt
pip install -e .
```

## 사용법

### M1: Baseline 측정

```bash
# KV 캐시 분석
python scripts/analyze_kv.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --context-length 8192

# Baseline 성능 측정
python scripts/run_baseline.py --context-length 8192
```

### M2: Level 1 압축 실험

```bash
# Level 1 압축 적용 inference
python experiments/m2_level1.py --context-length 32768 --page-size 128
```

### M3: Level 2 압축 실험

```bash
# Level 2 압축 비교
python experiments/m3_level2.py --compare-levels
```

## 프로젝트 구조

```
src/
├── models/              # 모델 로딩 및 inference 엔진
├── compression/         # 압축 알고리즘
│   ├── level1/         # GPU-native 압축
│   └── level2/         # CPU 엔트로피 코딩
├── cache/              # 페이지 관리 및 정책
├── analysis/           # KV 분석 도구
└── evaluation/         # 평가 메트릭
```

## 마일스톤

- [x] M1: Baseline 및 KV 분석 (Week 1-3)
- [ ] M2: Level 1 압축 설계 및 통합 (Week 3-6)
- [ ] M3: Level 2 실험 및 Ablation (Week 6-9)
- [ ] M4: 정리 및 논문화 (Week 9-12)

## 평가 지표

- **메모리**: KV 캐시 사용량 (무압축 대비 비율)
- **속도**: Tokens/s (Prefill & Decode)
- **품질**: Perplexity, QA 정확도, Long-context 벤치마크

## 라이선스

Apache-2.0

## 참고문헌

- SmolLM3: https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9
- vLLM Paged Attention: https://github.com/vllm-project/vllm
