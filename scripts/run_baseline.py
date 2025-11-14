#!/usr/bin/env python3
"""
Baseline 성능 측정 스크립트
- 무압축 KV 캐시로 inference 수행
- tokens/s, 메모리, latency 측정
"""

import sys
sys.path.insert(0, "/home/user/LCL/src")

import torch
import argparse
from models import SmolLMLoader, InferenceEngine
from evaluation import PerformanceMetrics, QualityMetrics
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Baseline Performance Measurement")
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=1024,
        help="Context length",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Max new tokens to generate",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs for averaging",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./baseline_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    # 디바이스 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 모델 로드
    logger.info("Loading model...")
    loader = SmolLMLoader(model_name=args.model, device=device)
    loader.load()

    # Inference 엔진
    engine = InferenceEngine(loader.model, loader.tokenizer, device=device)

    # 샘플 프롬프트
    prompts = [
        "Once upon a time, in a far away land, there was",
        "The future of artificial intelligence is",
        "In the world of technology, the most important innovation is",
    ]

    results = {
        "model": args.model,
        "context_length": args.context_length,
        "max_new_tokens": args.max_new_tokens,
        "num_runs": args.num_runs,
        "runs": [],
    }

    # 여러 번 실행
    for run_idx in range(args.num_runs):
        logger.info(f"\n=== Run {run_idx + 1}/{args.num_runs} ===")

        run_results = []

        for prompt_idx, prompt in enumerate(prompts):
            logger.info(f"\nPrompt {prompt_idx + 1}: {prompt[:50]}...")

            # 메모리 초기화
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

            # 생성
            generated_text, metrics = engine.generate(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=1.0,
                do_sample=False,
                collect_metrics=True,
            )

            logger.info(f"Generated: {generated_text[len(prompt):len(prompt)+100]}...")

            # 메트릭 출력
            logger.info(f"\nMetrics:")
            logger.info(f"  Prefill tokens/s: {metrics.prefill_tokens_per_sec:.2f}")
            logger.info(f"  Decode tokens/s: {metrics.decode_tokens_per_sec:.2f}")
            logger.info(f"  Total time: {metrics.total_time:.2f}s")
            logger.info(f"  KV memory: {PerformanceMetrics.format_memory(metrics.kv_memory_bytes)}")
            logger.info(f"  Peak memory: {PerformanceMetrics.format_memory(metrics.peak_memory_bytes)}")

            run_results.append({
                "prompt_idx": prompt_idx,
                "prefill_tokens_per_sec": metrics.prefill_tokens_per_sec,
                "decode_tokens_per_sec": metrics.decode_tokens_per_sec,
                "total_time": metrics.total_time,
                "kv_memory_bytes": metrics.kv_memory_bytes,
                "peak_memory_bytes": metrics.peak_memory_bytes,
            })

        results["runs"].append(run_results)

    # 평균 계산
    avg_metrics = {
        "avg_prefill_tokens_per_sec": 0.0,
        "avg_decode_tokens_per_sec": 0.0,
        "avg_total_time": 0.0,
        "avg_kv_memory_bytes": 0,
        "avg_peak_memory_bytes": 0,
    }

    total_samples = len(prompts) * args.num_runs

    for run in results["runs"]:
        for sample in run:
            avg_metrics["avg_prefill_tokens_per_sec"] += sample["prefill_tokens_per_sec"]
            avg_metrics["avg_decode_tokens_per_sec"] += sample["decode_tokens_per_sec"]
            avg_metrics["avg_total_time"] += sample["total_time"]
            avg_metrics["avg_kv_memory_bytes"] += sample["kv_memory_bytes"]
            avg_metrics["avg_peak_memory_bytes"] += sample["peak_memory_bytes"]

    for key in avg_metrics:
        avg_metrics[key] /= total_samples

    results["average"] = avg_metrics

    # 결과 저장
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n=== Average Results (over {total_samples} samples) ===")
    logger.info(f"Prefill tokens/s: {avg_metrics['avg_prefill_tokens_per_sec']:.2f}")
    logger.info(f"Decode tokens/s: {avg_metrics['avg_decode_tokens_per_sec']:.2f}")
    logger.info(f"Total time: {avg_metrics['avg_total_time']:.2f}s")
    logger.info(f"KV memory: {PerformanceMetrics.format_memory(int(avg_metrics['avg_kv_memory_bytes']))}")
    logger.info(f"Peak memory: {PerformanceMetrics.format_memory(int(avg_metrics['avg_peak_memory_bytes']))}")

    logger.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
