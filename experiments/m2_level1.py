#!/usr/bin/env python3
"""
M2: Level 1 압축 실험
- Transform + Quantization + Bit Packing
- 압축비, 속도, 품질 측정
"""

import sys
sys.path.insert(0, "/home/user/LCL/src")

import torch
import argparse
from models import SmolLMLoader, InferenceEngine
from compression import KVPageConfig, KVTransform, NearLosslessQuantizer, BitPacker
from analysis import KVAnalyzer
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_level1_compression(
    model_name: str,
    context_length: int,
    bits: int = 4,
    transform_type: str = "normalize",
    device: str = "cuda",
):
    """
    Level 1 압축 테스트

    Args:
        model_name: 모델 이름
        context_length: 컨텍스트 길이
        bits: 양자화 비트 수
        transform_type: Transform 타입
        device: 디바이스
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Level 1 Compression Test")
    logger.info(f"  bits={bits}, transform={transform_type}")
    logger.info(f"{'='*80}")

    # 모델 로드
    loader = SmolLMLoader(model_name=model_name, device=device)
    loader.load()

    engine = InferenceEngine(loader.model, loader.tokenizer, device=device)

    # 압축 컴포넌트
    transform = KVTransform(transform_type=transform_type, device=device)
    quantizer = NearLosslessQuantizer(device=device)
    packer = BitPacker(device=device)

    # 샘플 텍스트
    sample_text = "This is a test. " * (context_length // 4)

    inputs = loader.tokenizer(sample_text, return_tensors="pt")
    input_ids = inputs["input_ids"][:, :context_length].to(device)

    # Prefill
    with torch.no_grad():
        logits, past_key_values = engine.prefill(input_ids)

    # 원본 크기
    original_size = 0
    for k, v in past_key_values:
        original_size += k.numel() * k.element_size()
        original_size += v.numel() * v.element_size()

    logger.info(f"Original KV size: {loader.format_memory(original_size)}")

    # 압축 수행 (첫 번째 레이어만 테스트)
    k, v = past_key_values[0]

    # Shape: (batch, num_heads, seq_len, head_dim)
    # -> (num_heads, seq_len, head_dim)
    k = k[0]
    v = v[0]

    logger.info(f"\nK shape: {k.shape}")
    logger.info(f"V shape: {v.shape}")

    # Transform
    k_trans, k_meta = transform.transform(k)
    v_trans, v_meta = transform.transform(v)

    # Quantize
    k_quant, k_quant_meta = quantizer.quantize(k_trans, bits=bits, per_head=True)
    v_quant, v_quant_meta = quantizer.quantize(v_trans, bits=bits, per_head=True)

    # Pack
    k_packed = packer.pack(k_quant, bits=bits)
    v_packed = packer.pack(v_quant, bits=bits)

    # 압축 크기
    compressed_size = (
        k_packed.numel() * k_packed.element_size() +
        v_packed.numel() * v_packed.element_size()
    )

    # 메타데이터 크기 (간단히 추정)
    num_heads = k.shape[0]
    metadata_size = num_heads * 2 * 4 * 2  # scale/zero per head, K/V

    total_compressed = compressed_size + metadata_size

    compression_ratio = (k.numel() * 2 + v.numel() * 2) / total_compressed

    logger.info(f"\nCompressed size: {loader.format_memory(total_compressed)}")
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")

    # 복원
    k_unpacked = packer.unpack(k_packed, bits, k_quant.shape)
    v_unpacked = packer.unpack(v_packed, bits, v_quant.shape)

    k_dequant = quantizer.dequantize(k_unpacked, k_quant_meta)
    v_dequant = quantizer.dequantize(v_unpacked, v_quant_meta)

    k_recon = transform.inverse_transform(k_dequant, k_meta)
    v_recon = transform.inverse_transform(v_dequant, v_meta)

    # 복원 에러
    k_error = (k - k_recon).abs().mean().item()
    v_error = (v - v_recon).abs().mean().item()

    k_relative_error = (k_error / (k.abs().mean().item() + 1e-8))
    v_relative_error = (v_error / (v.abs().mean().item() + 1e-8))

    logger.info(f"\nReconstruction Error:")
    logger.info(f"  K MAE: {k_error:.6f} (relative: {k_relative_error:.2%})")
    logger.info(f"  V MAE: {v_error:.6f} (relative: {v_relative_error:.2%})")

    results = {
        "model": model_name,
        "context_length": context_length,
        "bits": bits,
        "transform_type": transform_type,
        "original_size_bytes": original_size,
        "compressed_size_bytes": int(total_compressed),
        "compression_ratio": float(compression_ratio),
        "k_mae": float(k_error),
        "v_mae": float(v_error),
        "k_relative_error": float(k_relative_error),
        "v_relative_error": float(v_relative_error),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="M2 Level 1 Compression Experiment")
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
        default=[4, 6, 8],
        help="Quantization bits to test",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        nargs="+",
        default=["none", "normalize", "differencing"],
        help="Transform types to test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./m2_level1_results.json",
        help="Output file",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    all_results = []

    # 모든 조합 테스트
    for bits in args.bits:
        for transform_type in args.transforms:
            try:
                result = test_level1_compression(
                    model_name=args.model,
                    context_length=args.context_length,
                    bits=bits,
                    transform_type=transform_type,
                    device=device,
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error with bits={bits}, transform={transform_type}: {e}")

    # 결과 저장
    with open(args.output, "w") as f:
        json.dump({"results": all_results}, f, indent=2)

    logger.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
