# Quick Start Guide

## 설치

```bash
# 저장소 클론 (이미 되어 있음)
cd /home/user/LCL

# 의존성 설치
pip install -r requirements.txt

# 패키지 설치
pip install -e .
```

## 기본 사용법

### 1. KV 캐시 분석

모델의 KV 캐시 특성을 분석합니다:

```bash
python scripts/analyze_kv.py \
  --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --context-length 1024 \
  --dump
```

출력:
- KV 캐시의 통계 (평균, 표준편차, 엔트로피 등)
- 압축 가능성 추정
- 메모리 사용량

### 2. Baseline 성능 측정

무압축 상태에서의 성능을 측정합니다:

```bash
python scripts/run_baseline.py \
  --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --context-length 1024 \
  --max-new-tokens 100 \
  --num-runs 3
```

측정 항목:
- Prefill tokens/s
- Decode tokens/s
- KV 메모리 사용량
- Peak 메모리 사용량

### 3. M1: 다양한 컨텍스트 길이 실험

```bash
python experiments/m1_baseline.py \
  --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --context-lengths 1024 2048 4096 \
  --output m1_results.json
```

### 4. M2: Level 1 압축 실험

Transform + Quantization + Bit Packing 압축 테스트:

```bash
python experiments/m2_level1.py \
  --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --context-length 1024 \
  --bits 4 6 8 \
  --transforms none normalize differencing \
  --output m2_results.json
```

### 5. M3: Level 2 압축 실험

엔트로피 코딩 추가 압축 테스트:

```bash
python experiments/m3_level2.py \
  --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --context-length 1024 \
  --bits 4 \
  --entropy-methods huffman rle simple \
  --output m3_results.json
```

## 결과 해석

### 압축비 (Compression Ratio)

- **Level 1**: 일반적으로 2~4x 압축
- **Level 2**: 추가로 1.2~2x 압축

### 복원 에러 (Reconstruction Error)

- **Near-lossless 목표**: 상대 에러 < 1%
- **Perplexity 증가**: < 3%

### 성능 (Performance)

- **Prefill**: 압축 오버헤드로 인해 약간 감소
- **Decode**: 페이지 복원 비용 추가
- **Trade-off**: 메모리 절감 vs 속도 저하

## 실험 예시

### 예시 1: 4bit 양자화 + Normalize Transform

```bash
python experiments/m2_level1.py \
  --bits 4 \
  --transforms normalize \
  --context-length 2048
```

예상 결과:
- 압축비: ~3.5x
- 상대 에러: 0.5~1.5%

### 예시 2: 엔트로피 코딩 비교

```bash
python experiments/m3_level2.py \
  --entropy-methods huffman rle simple
```

비교 항목:
- Huffman: 중간 압축비, 느림
- RLE: 낮은 압축비, 빠름
- Simple (zlib): 높은 압축비, 느림

## 다음 단계

1. **커스텀 데이터셋**: 자신의 데이터로 평가
2. **더 긴 컨텍스트**: 8k, 32k, 128k로 확장
3. **통합 Inference**: PageManager를 사용한 end-to-end 추론
4. **Ablation Study**: 각 컴포넌트의 기여도 분석

## 문제 해결

### CUDA Out of Memory

- 더 작은 컨텍스트 길이 사용
- 배치 크기 감소
- GPU 메모리가 큰 모델 사용

### 느린 속도

- Level 2는 CPU 기반이므로 느릴 수 있음
- Level 1만 사용 권장 (실시간 추론)
- 더 작은 모델 사용

## 지원

이슈가 있으면 GitHub Issues에 보고해주세요.
