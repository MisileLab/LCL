"""
Level 1 압축 - Transform 레이어 (GPU)
- KV 텐서의 분포를 좁혀 양자화를 용이하게 만듦
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class KVTransform:
    """
    KV 텐서 변환 레이어
    - GPU 상에서 동작
    - 분포를 좁혀 양자화 노이즈 최소화
    """

    def __init__(
        self,
        transform_type: str = "normalize",
        device: str = "cuda",
    ):
        """
        Args:
            transform_type: 변환 타입
                - "normalize": 채널/헤드별 정규화
                - "differencing": 인접 토큰 간 차분
                - "mixed": normalize + differencing
                - "none": 변환 없음
            device: 디바이스
        """
        self.transform_type = transform_type
        self.device = device

    def transform(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward transform

        Args:
            tensor: (num_heads, seq_len, head_dim)

        Returns:
            transformed_tensor, metadata
        """
        if self.transform_type == "none":
            return tensor, {"type": "none"}

        elif self.transform_type == "normalize":
            return self._normalize(tensor)

        elif self.transform_type == "differencing":
            return self._differencing(tensor)

        elif self.transform_type == "mixed":
            # normalize 후 differencing
            normalized, norm_meta = self._normalize(tensor)
            diffed, diff_meta = self._differencing(normalized)

            metadata = {
                "type": "mixed",
                "normalize": norm_meta,
                "differencing": diff_meta,
            }
            return diffed, metadata

        else:
            raise ValueError(f"Unknown transform type: {self.transform_type}")

    def inverse_transform(
        self,
        tensor: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Inverse transform (복원)

        Args:
            tensor: 변환된 텐서
            metadata: transform 시 저장된 메타데이터

        Returns:
            원본 텐서 (근사)
        """
        transform_type = metadata["type"]

        if transform_type == "none":
            return tensor

        elif transform_type == "normalize":
            return self._denormalize(tensor, metadata)

        elif transform_type == "differencing":
            return self._inverse_differencing(tensor, metadata)

        elif transform_type == "mixed":
            # 역순으로 복원
            tensor = self._inverse_differencing(tensor, metadata["differencing"])
            tensor = self._denormalize(tensor, metadata["normalize"])
            return tensor

        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

    def _normalize(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        헤드별 정규화
        - 각 헤드의 mean, std를 저장하고 정규화
        """
        # tensor shape: (num_heads, seq_len, head_dim)
        num_heads, seq_len, head_dim = tensor.shape

        # 헤드별 통계 (헤드마다 mean/std 계산)
        # Reshape to (num_heads, -1)
        tensor_flat = tensor.reshape(num_heads, -1)

        mean = tensor_flat.mean(dim=1, keepdim=True)  # (num_heads, 1)
        std = tensor_flat.std(dim=1, keepdim=True) + 1e-6  # (num_heads, 1)

        # 정규화
        normalized_flat = (tensor_flat - mean) / std
        normalized = normalized_flat.reshape(num_heads, seq_len, head_dim)

        metadata = {
            "type": "normalize",
            "mean": mean.cpu(),  # (num_heads, 1)
            "std": std.cpu(),    # (num_heads, 1)
        }

        return normalized, metadata

    def _denormalize(
        self,
        tensor: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> torch.Tensor:
        """정규화 복원"""
        num_heads, seq_len, head_dim = tensor.shape

        mean = metadata["mean"].to(tensor.device)
        std = metadata["std"].to(tensor.device)

        tensor_flat = tensor.reshape(num_heads, -1)
        denormalized_flat = tensor_flat * std + mean
        denormalized = denormalized_flat.reshape(num_heads, seq_len, head_dim)

        return denormalized

    def _differencing(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        인접 토큰 간 차분 (temporal differencing)
        - 시퀀스 방향으로 차분을 취함
        """
        # tensor shape: (num_heads, seq_len, head_dim)
        num_heads, seq_len, head_dim = tensor.shape

        if seq_len <= 1:
            # 차분 불가능
            return tensor, {"type": "differencing", "first_token": tensor[:, :1, :].cpu()}

        # 첫 번째 토큰은 그대로 저장 (차분 복원을 위해)
        first_token = tensor[:, :1, :].clone()

        # 차분 계산: diff[t] = tensor[t] - tensor[t-1]
        diff = torch.zeros_like(tensor)
        diff[:, 0, :] = tensor[:, 0, :]  # 첫 토큰은 그대로
        diff[:, 1:, :] = tensor[:, 1:, :] - tensor[:, :-1, :]

        metadata = {
            "type": "differencing",
            "first_token": first_token.cpu(),
        }

        return diff, metadata

    def _inverse_differencing(
        self,
        tensor: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> torch.Tensor:
        """차분 복원 (cumulative sum)"""
        # tensor shape: (num_heads, seq_len, head_dim)
        # cumsum을 통해 원본 복원

        # 시퀀스 방향으로 누적합
        reconstructed = torch.cumsum(tensor, dim=1)

        return reconstructed
