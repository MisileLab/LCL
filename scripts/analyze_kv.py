#!/usr/bin/env python3
"""
KV 캐시 분석 스크립트
- 모델 로드 및 샘플 텍스트에 대한 KV 캐시 생성
- KV 캐시의 통계, 분포, 엔트로피 분석
"""

import torch
import argparse
import logging

from llm_kv_zram.models import SmolLMLoader, InferenceEngine
from llm_kv_zram.analysis import KVAnalyzer, KVDumper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="KV Cache Analysis")
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
        help="Context length for analysis",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./kv_analysis_output",
        help="Output directory",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump KV cache to file",
    )

    args = parser.parse_args()

    # 디바이스 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 모델 로드
    logger.info("Loading model...")
    loader = SmolLMLoader(model_name=args.model, device=device)
    loader.load()

    # 모델 정보 출력
    info = loader.get_model_info()
    logger.info(f"Model info: {info}")

    # 예상 메모리 사용량
    mem_estimate = loader.estimate_kv_memory(args.context_length)
    logger.info("\n=== Estimated KV Memory Usage ===")
    for dtype, size in mem_estimate.items():
        if dtype != "num_elements":
            logger.info(f"{dtype}: {loader.format_memory(size)}")

    # Inference 엔진
    engine = InferenceEngine(loader.model, loader.tokenizer, device=device)

    # 샘플 텍스트 (긴 컨텍스트)
    sample_text = (
        "Natural language processing (NLP) is a subfield of linguistics, computer science, "
        "and artificial intelligence concerned with the interactions between computers and human language. "
    ) * (args.context_length // 100)

    # Prefill 수행 (KV 캐시 생성)
    logger.info("\n=== Generating KV Cache ===")
    inputs = loader.tokenizer(sample_text, return_tensors="pt")
    input_ids = inputs["input_ids"][:, :args.context_length]

    logger.info(f"Input tokens: {input_ids.shape[1]}")

    with torch.no_grad():
        logits, past_key_values = engine.prefill(input_ids)

    logger.info("KV cache generated successfully")

    # KV 캐시 분석
    logger.info("\n=== Analyzing KV Cache ===")
    analyzer = KVAnalyzer(device=device)

    analysis = analyzer.analyze_kv_cache(past_key_values, detailed=True)

    # 분석 결과 출력
    report = analyzer.format_analysis_report(analysis)
    print(report)

    # 압축 가능성 추정
    logger.info("\n=== Compression Potential ===")
    for bits in [4, 6, 8]:
        compression_est = analyzer.estimate_compression_potential(
            past_key_values,
            target_bits=bits,
        )
        logger.info(f"\n{bits}-bit quantization:")
        logger.info(f"  Current memory: {loader.format_memory(compression_est['current_memory_bytes'])}")
        logger.info(f"  Estimated compressed: {loader.format_memory(compression_est['estimated_compressed_bytes'])}")
        logger.info(f"  Compression ratio: {compression_est['compression_ratio']:.2f}x")

    # Dump (옵션)
    if args.dump:
        logger.info("\n=== Dumping KV Cache ===")
        dumper = KVDumper(output_dir=args.output_dir)
        dumper.dump_for_analysis(
            past_key_values,
            analysis,
            prefix=f"analysis_{args.context_length}",
        )

    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()
