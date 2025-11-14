#!/usr/bin/env python3
"""
M1: Baseline 실험
- 무압축 KV 캐시로 다양한 컨텍스트 길이에서 성능 측정
"""

import sys
sys.path.insert(0, "/home/user/LCL/src")

import torch
import argparse
from models import SmolLMLoader, InferenceEngine
from analysis import KVAnalyzer
from evaluation import PerformanceMetrics, QualityMetrics
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_baseline_experiment(
    model_name: str,
    context_lengths: list,
    device: str = "cuda",
):
    """
    Baseline 실험 실행

    Args:
        model_name: 모델 이름
        context_lengths: 테스트할 컨텍스트 길이 리스트
        device: 디바이스
    """
    results = {
        "model": model_name,
        "experiments": [],
    }

    # 모델 로드
    logger.info("Loading model...")
    loader = SmolLMLoader(model_name=model_name, device=device)
    loader.load()

    engine = InferenceEngine(loader.model, loader.tokenizer, device=device)
    analyzer = KVAnalyzer(device=device)

    # 각 컨텍스트 길이에 대해 실험
    for ctx_len in context_lengths:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing context length: {ctx_len}")
        logger.info(f"{'='*80}")

        # 샘플 텍스트 생성
        sample_text = (
            "This is a sample text for testing KV cache compression. " * (ctx_len // 10)
        )

        # 토큰화
        inputs = loader.tokenizer(sample_text, return_tensors="pt")
        input_ids = inputs["input_ids"][:, :ctx_len].to(device)

        actual_length = input_ids.shape[1]
        logger.info(f"Actual token count: {actual_length}")

        # 메모리 초기화
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Prefill 수행
        with torch.no_grad():
            logits, past_key_values = engine.prefill(input_ids)

        # KV 캐시 분석
        analysis = analyzer.analyze_kv_cache(past_key_values, detailed=True)

        # 메모리 측정
        memory_stats = PerformanceMetrics.measure_memory_usage(device)

        # 결과 저장
        exp_result = {
            "context_length": ctx_len,
            "actual_tokens": actual_length,
            "kv_total_memory_bytes": analysis["total_memory_bytes"],
            "kv_total_memory_formatted": PerformanceMetrics.format_memory(
                analysis["total_memory_bytes"]
            ),
            "peak_memory_bytes": memory_stats["max_allocated"],
            "peak_memory_formatted": PerformanceMetrics.format_memory(
                memory_stats["max_allocated"]
            ),
            "avg_k_entropy": sum(
                layer["k_entropy"] for layer in analysis["layers"]
            ) / len(analysis["layers"]),
            "avg_v_entropy": sum(
                layer["v_entropy"] for layer in analysis["layers"]
            ) / len(analysis["layers"]),
        }

        results["experiments"].append(exp_result)

        # 로그 출력
        logger.info(f"\nResults:")
        logger.info(f"  KV memory: {exp_result['kv_total_memory_formatted']}")
        logger.info(f"  Peak memory: {exp_result['peak_memory_formatted']}")
        logger.info(f"  Avg K entropy: {exp_result['avg_k_entropy']:.2f}")
        logger.info(f"  Avg V entropy: {exp_result['avg_v_entropy']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="M1 Baseline Experiment")
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096],
        help="Context lengths to test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./m1_baseline_results.json",
        help="Output file",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 실험 실행
    results = run_baseline_experiment(
        model_name=args.model,
        context_lengths=args.context_lengths,
        device=device,
    )

    # 결과 저장
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
