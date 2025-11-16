#!/usr/bin/env python3
"""
M3: Level 2 압축 실험 (엔트로피 코딩)
- Level 1 + Level 2 압축비 비교
- CPU 기반 오프라인 압축
"""

import sys
import os
from pathlib import Path

# Add src directory to path
# Try multiple methods to find the project root
script_dir = Path(__file__).resolve().parent  # resolve() handles symlinks
project_root = script_dir.parent
src_dir = project_root / "src"

# Fallback: try current working directory
if not src_dir.exists():
    cwd_src = Path.cwd() / "src"
    if cwd_src.exists():
        src_dir = cwd_src
        project_root = Path.cwd()

sys.path.insert(0, str(src_dir))

import torch
import argparse
from models import SmolLMLoader, InferenceEngine
from compression import KVTransform, NearLosslessQuantizer, BitPacker, EntropyCoder
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_level2_compression(
    model_name: str,
    context_length: int,
    bits: int = 4,
    transform_type: str = "normalize",
    entropy_method: str = "huffman",
    device: str = "cuda",
):
    """
    Level 2 압축 테스트 (Level 1 + 엔트로피 코딩)

    Args:
        model_name: 모델 이름
        context_length: 컨텍스트 길이
        bits: 양자화 비트 수
        transform_type: Transform 타입
        entropy_method: 엔트로피 코딩 방법
        device: 디바이스
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Level 2 Compression Test")
    logger.info(f"  bits={bits}, transform={transform_type}, entropy={entropy_method}")
    logger.info(f"{'='*80}")

    # 모델 로드
    loader = SmolLMLoader(model_name=model_name, device=device)
    loader.load()

    engine = InferenceEngine(loader.model, loader.tokenizer, device=device)

    # 압축 컴포넌트
    transform = KVTransform(transform_type=transform_type, device=device)
    quantizer = NearLosslessQuantizer(device=device)
    packer = BitPacker(device=device)
    entropy_coder = EntropyCoder()

    # 샘플 텍스트
    sample_text = "This is a test. " * (context_length // 4)

    inputs = loader.tokenizer(sample_text, return_tensors="pt")
    input_ids = inputs["input_ids"][:, :context_length].to(device)

    # Prefill
    with torch.no_grad():
        logits, past_key_values = engine.prefill(input_ids)

    # 원본 크기
    k, v = past_key_values[0]
    k = k[0]
    v = v[0]

    original_size = (k.numel() + v.numel()) * 2  # bf16/fp16 = 2 bytes

    logger.info(f"Original size: {loader.format_memory(original_size)}")

    # Level 1 압축
    k_trans, k_meta = transform.transform(k)
    v_trans, v_meta = transform.transform(v)

    k_quant, k_quant_meta = quantizer.quantize(k_trans, bits=bits, per_head=True)
    v_quant, v_quant_meta = quantizer.quantize(v_trans, bits=bits, per_head=True)

    k_packed = packer.pack(k_quant, bits=bits)
    v_packed = packer.pack(v_quant, bits=bits)

    level1_size = (
        k_packed.numel() * k_packed.element_size() +
        v_packed.numel() * v_packed.element_size()
    )

    logger.info(f"Level 1 size: {loader.format_memory(level1_size)}")
    logger.info(f"Level 1 ratio: {original_size / level1_size:.2f}x")

    # Level 2 압축 (엔트로피 코딩)
    k_entropy, k_entropy_meta = entropy_coder.encode(k_packed.cpu(), method=entropy_method)
    v_entropy, v_entropy_meta = entropy_coder.encode(v_packed.cpu(), method=entropy_method)

    level2_size = len(k_entropy) + len(v_entropy)

    logger.info(f"\nLevel 2 size: {loader.format_memory(level2_size)}")
    logger.info(f"Level 2 ratio (vs original): {original_size / level2_size:.2f}x")
    logger.info(f"Level 2 ratio (vs Level 1): {level1_size / level2_size:.2f}x")

    # 복원 테스트
    k_decoded = entropy_coder.decode(k_entropy, k_entropy_meta)
    v_decoded = entropy_coder.decode(v_entropy, v_entropy_meta)

    # 검증
    k_match = torch.equal(k_packed.cpu(), k_decoded)
    v_match = torch.equal(v_packed.cpu(), v_decoded)

    logger.info(f"\nReconstruction check:")
    logger.info(f"  K match: {k_match}")
    logger.info(f"  V match: {v_match}")

    results = {
        "model": model_name,
        "context_length": context_length,
        "bits": bits,
        "transform_type": transform_type,
        "entropy_method": entropy_method,
        "original_size_bytes": int(original_size),
        "level1_size_bytes": int(level1_size),
        "level2_size_bytes": int(level2_size),
        "level1_ratio": float(original_size / level1_size),
        "level2_ratio": float(original_size / level2_size),
        "level2_vs_level1_ratio": float(level1_size / level2_size),
        "k_match": k_match,
        "v_match": v_match,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="M3 Level 2 Compression Experiment")
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
        "--bits",
        type=int,
        nargs="+",
        default=[4],
        help="Quantization bits to test",
    )
    parser.add_argument(
        "--entropy-methods",
        type=str,
        nargs="+",
        default=["huffman", "rle", "simple"],
        help="Entropy coding methods",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./m3_level2_results.json",
        help="Output file",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    all_results = []

    # 모든 조합 테스트
    for bits in args.bits:
        for entropy_method in args.entropy_methods:
            try:
                result = test_level2_compression(
                    model_name=args.model,
                    context_length=args.context_length,
                    bits=bits,
                    transform_type="normalize",
                    entropy_method=entropy_method,
                    device=device,
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error with bits={bits}, entropy={entropy_method}: {e}")

    # 결과 저장
    with open(args.output, "w") as f:
        json.dump({"results": all_results}, f, indent=2)

    logger.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
