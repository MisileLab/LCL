"""
KV 캐시 분석 도구
- 분포, 통계, 엔트로피 분석
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class KVAnalyzer:
    """KV 캐시의 통계적 특성 분석"""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def analyze_kv_cache(
        self,
        past_key_values: Tuple,
        detailed: bool = True,
    ) -> Dict[str, Any]:
        """
        KV 캐시의 전반적인 통계 분석

        Args:
            past_key_values: 모델의 KV 캐시
            detailed: 상세 분석 여부

        Returns:
            분석 결과 딕셔너리
        """
        if past_key_values is None:
            return {}

        results = {
            "num_layers": len(past_key_values),
            "layers": [],
        }

        total_memory = 0
        all_k_stats = []
        all_v_stats = []

        for layer_idx, (k, v) in enumerate(past_key_values):
            layer_stats = {
                "layer_idx": layer_idx,
                "k_shape": tuple(k.shape),
                "v_shape": tuple(v.shape),
            }

            # K 통계
            k_stats = self._compute_tensor_stats(k)
            layer_stats["k_stats"] = k_stats
            all_k_stats.append(k_stats)

            # V 통계
            v_stats = self._compute_tensor_stats(v)
            layer_stats["v_stats"] = v_stats
            all_v_stats.append(v_stats)

            # 메모리
            k_memory = k.numel() * k.element_size()
            v_memory = v.numel() * v.element_size()
            layer_stats["k_memory_bytes"] = k_memory
            layer_stats["v_memory_bytes"] = v_memory
            layer_stats["total_memory_bytes"] = k_memory + v_memory

            total_memory += k_memory + v_memory

            # 상세 분석
            if detailed:
                layer_stats["k_entropy"] = self._estimate_entropy(k)
                layer_stats["v_entropy"] = self._estimate_entropy(v)
                layer_stats["k_sparsity"] = self._compute_sparsity(k)
                layer_stats["v_sparsity"] = self._compute_sparsity(v)

            results["layers"].append(layer_stats)

        # 전체 통계
        results["total_memory_bytes"] = total_memory
        results["avg_k_stats"] = self._average_stats(all_k_stats)
        results["avg_v_stats"] = self._average_stats(all_v_stats)

        return results

    def _compute_tensor_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        """텐서의 기본 통계 계산"""
        tensor_flat = tensor.flatten().float()

        return {
            "mean": tensor_flat.mean().item(),
            "std": tensor_flat.std().item(),
            "min": tensor_flat.min().item(),
            "max": tensor_flat.max().item(),
            "abs_mean": tensor_flat.abs().mean().item(),
            "abs_max": tensor_flat.abs().max().item(),
        }

    def _estimate_entropy(self, tensor: torch.Tensor, num_bins: int = 256) -> float:
        """
        텐서의 엔트로피 추정 (히스토그램 기반)

        높은 엔트로피 = 더 다양한 값 분포 = 압축 어려움
        """
        tensor_flat = tensor.flatten().float().cpu()

        # 히스토그램 계산
        hist, _ = np.histogram(tensor_flat.numpy(), bins=num_bins, density=True)

        # 확률로 변환 (0 제거)
        hist = hist[hist > 0]
        hist = hist / hist.sum()

        # 엔트로피 계산: H(X) = -Σ p(x) log2(p(x))
        entropy = -np.sum(hist * np.log2(hist))

        return float(entropy)

    def _compute_sparsity(self, tensor: torch.Tensor, threshold: float = 1e-6) -> float:
        """
        텐서의 희소성 계산

        Returns:
            0에 가까운 값들의 비율 (0~1)
        """
        tensor_flat = tensor.flatten().abs()
        near_zero = (tensor_flat < threshold).sum().item()
        total = tensor_flat.numel()

        return near_zero / total

    def _average_stats(self, stats_list: list) -> Dict[str, float]:
        """통계 리스트의 평균 계산"""
        if not stats_list:
            return {}

        keys = stats_list[0].keys()
        avg_stats = {}

        for key in keys:
            values = [s[key] for s in stats_list]
            avg_stats[key] = np.mean(values)

        return avg_stats

    def analyze_distribution_by_position(
        self,
        past_key_values: Tuple,
        layer_idx: int = 0,
        head_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        특정 레이어/헤드의 위치별 분포 분석
        (token position에 따른 값 변화 패턴 확인)
        """
        k, v = past_key_values[layer_idx]

        # Shape: (batch, num_heads, seq_len, head_dim)
        batch_size, num_heads, seq_len, head_dim = k.shape

        if head_idx >= num_heads:
            head_idx = 0

        # 특정 헤드 선택
        k_head = k[0, head_idx, :, :].cpu()  # (seq_len, head_dim)
        v_head = v[0, head_idx, :, :].cpu()  # (seq_len, head_dim)

        # 위치별 통계
        position_stats = {
            "layer_idx": layer_idx,
            "head_idx": head_idx,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "k_norm_by_position": [],
            "v_norm_by_position": [],
        }

        for pos in range(seq_len):
            k_norm = torch.norm(k_head[pos]).item()
            v_norm = torch.norm(v_head[pos]).item()

            position_stats["k_norm_by_position"].append(k_norm)
            position_stats["v_norm_by_position"].append(v_norm)

        return position_stats

    def analyze_head_importance(
        self,
        past_key_values: Tuple,
        layer_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        헤드별 중요도 분석 (L2 norm 기반)
        """
        k, v = past_key_values[layer_idx]

        # Shape: (batch, num_heads, seq_len, head_dim)
        batch_size, num_heads, seq_len, head_dim = k.shape

        head_importance = {
            "layer_idx": layer_idx,
            "num_heads": num_heads,
            "k_norms": [],
            "v_norms": [],
        }

        for head_idx in range(num_heads):
            k_head = k[0, head_idx, :, :]
            v_head = v[0, head_idx, :, :]

            k_norm = torch.norm(k_head).item()
            v_norm = torch.norm(v_head).item()

            head_importance["k_norms"].append(k_norm)
            head_importance["v_norms"].append(v_norm)

        return head_importance

    def estimate_compression_potential(
        self,
        past_key_values: Tuple,
        target_bits: int = 4,
    ) -> Dict[str, Any]:
        """
        압축 가능성 추정

        Args:
            past_key_values: KV 캐시
            target_bits: 목표 비트수 (4bit, 8bit 등)

        Returns:
            압축비 및 예상 오차 정보
        """
        results = self.analyze_kv_cache(past_key_values, detailed=True)

        # 현재 메모리
        current_memory = results["total_memory_bytes"]

        # 압축 후 예상 메모리 (scale/zero-point 오버헤드 포함)
        num_layers = results["num_layers"]
        first_layer = results["layers"][0]
        k_shape = first_layer["k_shape"]

        # (batch, num_heads, seq_len, head_dim)
        num_elements = np.prod(k_shape) * 2 * num_layers  # K + V

        # Target bits로 압축
        compressed_data = num_elements * target_bits / 8

        # 메타데이터 오버헤드 (scale, zero-point per head per layer)
        num_heads = k_shape[1]
        metadata_overhead = num_layers * num_heads * 2 * 4  # float32 scale/zero

        estimated_memory = compressed_data + metadata_overhead

        compression_ratio = current_memory / estimated_memory

        return {
            "current_memory_bytes": current_memory,
            "estimated_compressed_bytes": estimated_memory,
            "compression_ratio": compression_ratio,
            "target_bits": target_bits,
            "avg_entropy": np.mean([
                layer["k_entropy"] for layer in results["layers"]
            ]),
        }

    def format_analysis_report(self, analysis: Dict[str, Any]) -> str:
        """분석 결과를 human-readable 형식으로 포맷"""
        lines = []
        lines.append("=" * 80)
        lines.append("KV Cache Analysis Report")
        lines.append("=" * 80)

        lines.append(f"\nNumber of layers: {analysis['num_layers']}")
        lines.append(f"Total memory: {self._format_bytes(analysis['total_memory_bytes'])}")

        lines.append("\n--- Average K Statistics ---")
        for key, val in analysis["avg_k_stats"].items():
            lines.append(f"  {key}: {val:.6f}")

        lines.append("\n--- Average V Statistics ---")
        for key, val in analysis["avg_v_stats"].items():
            lines.append(f"  {key}: {val:.6f}")

        # 레이어별 요약
        if "layers" in analysis and len(analysis["layers"]) > 0:
            lines.append("\n--- Per-Layer Summary ---")
            for layer in analysis["layers"][:3]:  # 처음 3개만
                lines.append(f"\nLayer {layer['layer_idx']}:")
                lines.append(f"  Shape: {layer['k_shape']}")
                lines.append(f"  Memory: {self._format_bytes(layer['total_memory_bytes'])}")
                if "k_entropy" in layer:
                    lines.append(f"  K Entropy: {layer['k_entropy']:.2f}")
                    lines.append(f"  V Entropy: {layer['v_entropy']:.2f}")
                if "k_sparsity" in layer:
                    lines.append(f"  K Sparsity: {layer['k_sparsity']:.2%}")
                    lines.append(f"  V Sparsity: {layer['v_sparsity']:.2%}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def _format_bytes(self, bytes_size: float) -> str:
        """바이트 크기를 human-readable 형식으로"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} TB"
