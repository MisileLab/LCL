"""
Level 1 압축 - Near-lossless Quantization (GPU)
- 3~4bit mixed-precision 양자화
- Per-head scale/zero-point
"""

import torch
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class NearLosslessQuantizer:
    """
    Near-lossless 양자화기
    - GPU 상에서 동작
    - Per-head scale/zero-point 지원
    - Mixed-precision 지원 (레이어/헤드별 중요도)
    """

    def __init__(
        self,
        device: str = "cuda",
        use_symmetric: bool = True,
    ):
        """
        Args:
            device: 디바이스
            use_symmetric: 대칭 양자화 사용 (zero-point = 0)
        """
        self.device = device
        self.use_symmetric = use_symmetric

    def quantize(
        self,
        tensor: torch.Tensor,
        bits: int = 4,
        per_head: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        양자화

        Args:
            tensor: (num_heads, seq_len, head_dim)
            bits: 양자화 비트 수 (3, 4, 6, 8)
            per_head: 헤드별 scale/zero-point

        Returns:
            quantized_tensor (int), metadata
        """
        num_heads, seq_len, head_dim = tensor.shape

        # 양자화 범위
        qmin = 0
        qmax = 2 ** bits - 1

        if per_head:
            # 헤드별로 scale/zero-point 계산
            scale, zero_point = self._compute_scale_zero_per_head(
                tensor, qmin, qmax
            )
        else:
            # 전체 텐서에 대해 하나의 scale/zero-point
            scale, zero_point = self._compute_scale_zero_global(
                tensor, qmin, qmax
            )

        # 양자화: q = round((x - zero) / scale)
        # 단, 우리는 scale/zero를 이미 계산했으므로
        # q = clamp(round(x / scale + zero_point), qmin, qmax)
        quantized = torch.clamp(
            torch.round(tensor / scale + zero_point),
            qmin,
            qmax
        ).to(torch.int32)

        metadata = {
            "bits": bits,
            "scale": scale.cpu(),
            "zero_point": zero_point.cpu(),
            "qmin": qmin,
            "qmax": qmax,
            "per_head": per_head,
            "shape": tuple(tensor.shape),
        }

        return quantized, metadata

    def dequantize(
        self,
        quantized: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> torch.Tensor:
        """
        역양자화

        Args:
            quantized: 양자화된 텐서 (int32)
            metadata: 양자화 시 저장된 메타데이터

        Returns:
            dequantized_tensor (float)
        """
        scale = metadata["scale"].to(quantized.device)
        zero_point = metadata["zero_point"].to(quantized.device)

        # 역양자화: x = (q - zero_point) * scale
        dequantized = (quantized.float() - zero_point) * scale

        return dequantized

    def _compute_scale_zero_per_head(
        self,
        tensor: torch.Tensor,
        qmin: int,
        qmax: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        헤드별 scale/zero-point 계산

        Returns:
            scale: (num_heads, 1, 1)
            zero_point: (num_heads, 1, 1)
        """
        num_heads, seq_len, head_dim = tensor.shape

        # 헤드별로 min/max 계산
        # Reshape to (num_heads, -1)
        tensor_per_head = tensor.reshape(num_heads, -1)

        tmin = tensor_per_head.min(dim=1, keepdim=True)[0]  # (num_heads, 1)
        tmax = tensor_per_head.max(dim=1, keepdim=True)[0]  # (num_heads, 1)

        if self.use_symmetric:
            # 대칭 양자화: abs_max 사용
            abs_max = torch.max(tmin.abs(), tmax.abs())
            scale = abs_max / (qmax / 2)
            zero_point = torch.zeros_like(scale) + (qmax / 2)
        else:
            # 비대칭 양자화
            scale = (tmax - tmin) / (qmax - qmin)
            zero_point = qmin - tmin / scale

        # Avoid division by zero
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        # Reshape to (num_heads, 1, 1)
        scale = scale.unsqueeze(-1)
        zero_point = zero_point.unsqueeze(-1)

        return scale, zero_point

    def _compute_scale_zero_global(
        self,
        tensor: torch.Tensor,
        qmin: int,
        qmax: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        전체 텐서에 대해 단일 scale/zero-point 계산

        Returns:
            scale: scalar
            zero_point: scalar
        """
        tmin = tensor.min()
        tmax = tensor.max()

        if self.use_symmetric:
            abs_max = max(tmin.abs(), tmax.abs())
            scale = abs_max / (qmax / 2)
            zero_point = torch.tensor(qmax / 2, device=tensor.device)
        else:
            scale = (tmax - tmin) / (qmax - qmin)
            zero_point = qmin - tmin / scale

        if scale == 0:
            scale = torch.tensor(1.0, device=tensor.device)

        return scale, zero_point

    def estimate_quantization_error(
        self,
        tensor: torch.Tensor,
        bits: int = 4,
        per_head: bool = True,
    ) -> Dict[str, float]:
        """
        양자화 에러 추정 (양자화 -> 역양자화 후 차이)

        Returns:
            에러 통계 딕셔너리
        """
        quantized, metadata = self.quantize(tensor, bits, per_head)
        dequantized = self.dequantize(quantized, metadata)

        error = (tensor - dequantized).abs()

        return {
            "mean_abs_error": error.mean().item(),
            "max_abs_error": error.max().item(),
            "mse": (error ** 2).mean().item(),
            "relative_error": (error / (tensor.abs() + 1e-8)).mean().item(),
        }

    def adaptive_bit_allocation(
        self,
        tensor: torch.Tensor,
        importance_scores: torch.Tensor,
        bit_budget: int = 4,
    ) -> torch.Tensor:
        """
        헤드별 중요도에 따른 적응적 비트 할당
        (Mixed-precision 지원)

        Args:
            tensor: (num_heads, seq_len, head_dim)
            importance_scores: (num_heads,) - 헤드별 중요도 점수
            bit_budget: 평균 비트 수

        Returns:
            bits_per_head: (num_heads,) - 헤드별 할당된 비트 수
        """
        num_heads = tensor.shape[0]

        # 중요도를 [0, 1]로 정규화
        importance_norm = (importance_scores - importance_scores.min()) / (
            importance_scores.max() - importance_scores.min() + 1e-8
        )

        # 비트 할당: 중요한 헤드에 더 많은 비트
        # 간단한 전략: importance에 비례하여 3~8bit 할당
        min_bits = 3
        max_bits = 8

        bits_per_head = (
            min_bits + importance_norm * (max_bits - min_bits)
        ).round().long()

        # 평균 비트가 bit_budget이 되도록 조정
        current_avg = bits_per_head.float().mean()
        adjustment = bit_budget - current_avg

        # 간단한 조정: 전체에 균등하게 더하거나 빼기
        bits_per_head = (bits_per_head.float() + adjustment).round().long()
        bits_per_head = torch.clamp(bits_per_head, min_bits, max_bits)

        return bits_per_head
