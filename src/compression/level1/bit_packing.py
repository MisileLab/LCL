"""
Level 1 압축 - Bit Packing/Unpacking (GPU)
- 고정길이 bit packing
- GPU 친화적 구현
"""

import torch
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class BitPacker:
    """
    Bit packing/unpacking 유틸리티
    - GPU 상에서 동작
    - 고정길이 packing (예: 4bit 값 8개 -> 32bit 워드)
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def pack(
        self,
        quantized: torch.Tensor,
        bits: int = 4,
    ) -> torch.Tensor:
        """
        양자화된 값들을 bit packing

        Args:
            quantized: (num_heads, seq_len, head_dim) int32 텐서
            bits: 비트 수 (3, 4, 6, 8)

        Returns:
            packed: 패킹된 텐서
        """
        if bits == 8:
            # 8bit는 packing 불필요
            return quantized.to(torch.uint8)

        # 4bit packing 예시: 2개 값을 1바이트에 패킹
        if bits == 4:
            return self._pack_4bit(quantized)
        elif bits == 3:
            # 3bit는 더 복잡하지만 간단히 4bit로 처리
            logger.warning("3bit packing is approximated as 4bit")
            return self._pack_4bit(quantized)
        elif bits == 6:
            # 6bit는 간단히 8bit로 처리 (향후 최적화 가능)
            logger.warning("6bit packing is approximated as 8bit")
            return quantized.to(torch.uint8)
        else:
            raise ValueError(f"Unsupported bits: {bits}")

    def unpack(
        self,
        packed: torch.Tensor,
        bits: int,
        original_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        패킹된 텐서를 복원

        Args:
            packed: 패킹된 텐서
            bits: 비트 수
            original_shape: 원본 shape

        Returns:
            unpacked: (original_shape) int32 텐서
        """
        if bits == 8:
            return packed.to(torch.int32).reshape(original_shape)

        if bits == 4 or bits == 3:
            return self._unpack_4bit(packed, original_shape)
        elif bits == 6:
            return packed.to(torch.int32).reshape(original_shape)
        else:
            raise ValueError(f"Unsupported bits: {bits}")

    def _pack_4bit(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        4bit packing: 2개 값을 1바이트에 패킹

        Args:
            quantized: int32 텐서

        Returns:
            packed: uint8 텐서 (원본 크기의 절반)
        """
        # Flatten
        flat = quantized.flatten()

        # 패딩 (짝수 개수로 맞춤)
        if flat.numel() % 2 != 0:
            flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype, device=flat.device)])

        # 4bit로 clamp
        flat = torch.clamp(flat, 0, 15)

        # 2개씩 묶어서 1바이트로
        # high nibble (상위 4bit), low nibble (하위 4bit)
        even = flat[0::2]  # 짝수 인덱스
        odd = flat[1::2]   # 홀수 인덱스

        packed = ((even << 4) | odd).to(torch.uint8)

        return packed

    def _unpack_4bit(
        self,
        packed: torch.Tensor,
        original_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        4bit unpacking

        Args:
            packed: uint8 텐서
            original_shape: 원본 shape

        Returns:
            unpacked: int32 텐서
        """
        # 각 바이트를 2개 값으로 분해
        high = (packed >> 4).to(torch.int32)  # 상위 4bit
        low = (packed & 0x0F).to(torch.int32)  # 하위 4bit

        # Interleave
        unpacked = torch.stack([high, low], dim=-1).flatten()

        # 원본 크기로 자르기 (패딩 제거)
        total_elements = 1
        for dim in original_shape:
            total_elements *= dim

        unpacked = unpacked[:total_elements]

        # Reshape
        unpacked = unpacked.reshape(original_shape)

        return unpacked

    def estimate_packed_size(
        self,
        original_shape: Tuple[int, ...],
        bits: int,
    ) -> int:
        """
        패킹 후 예상 크기 (bytes)

        Args:
            original_shape: 원본 shape
            bits: 비트 수

        Returns:
            예상 바이트 크기
        """
        num_elements = 1
        for dim in original_shape:
            num_elements *= dim

        if bits == 8:
            return num_elements
        elif bits == 4 or bits == 3:
            # 2개당 1바이트
            return (num_elements + 1) // 2
        elif bits == 6:
            # 간단히 8bit로 근사
            return num_elements
        else:
            return num_elements * bits // 8
