"""
KV 페이지 포맷 정의
- OS의 페이지와 유사하게, KV 캐시를 페이지 단위로 관리
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class KVPageConfig:
    """KV 페이지 설정"""
    page_size: int = 128  # 페이지당 토큰 수
    bits_per_element: int = 4  # 양자화 비트 수
    use_transform: bool = True  # Transform 사용 여부
    transform_type: str = "normalize"  # normalize, differencing, none
    per_head_scale: bool = True  # 헤드별 scale/zero-point
    device: str = "cuda"


class KVPage:
    """
    KV 캐시 페이지
    - 고정된 토큰 범위를 하나의 압축 단위로 관리
    """

    def __init__(
        self,
        layer_idx: int,
        page_idx: int,
        token_start: int,
        token_end: int,
        config: KVPageConfig,
    ):
        """
        Args:
            layer_idx: 레이어 인덱스
            page_idx: 페이지 인덱스
            token_start: 시작 토큰 위치 (inclusive)
            token_end: 끝 토큰 위치 (exclusive)
            config: 페이지 설정
        """
        self.layer_idx = layer_idx
        self.page_idx = page_idx
        self.token_start = token_start
        self.token_end = token_end
        self.config = config

        # 압축된 데이터 저장소
        self.compressed_k: Optional[torch.Tensor] = None
        self.compressed_v: Optional[torch.Tensor] = None

        # 메타데이터 (scale, zero-point 등)
        self.k_metadata: Optional[dict] = None
        self.v_metadata: Optional[dict] = None

        # 상태
        self.is_compressed = False
        self.is_hot = False  # Hot cache에 있는지 여부

    @property
    def num_tokens(self) -> int:
        """페이지 내 토큰 수"""
        return self.token_end - self.token_start

    def compress(
        self,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        transform_fn,
        quantize_fn,
        pack_fn,
    ):
        """
        K, V 텐서를 압축

        Args:
            k_tensor: (num_heads, num_tokens, head_dim)
            v_tensor: (num_heads, num_tokens, head_dim)
            transform_fn: Transform 함수
            quantize_fn: Quantization 함수
            pack_fn: Bit packing 함수
        """
        # Transform
        k_transformed, k_transform_meta = transform_fn(k_tensor)
        v_transformed, v_transform_meta = transform_fn(v_tensor)

        # Quantize
        k_quantized, k_quant_meta = quantize_fn(
            k_transformed,
            bits=self.config.bits_per_element,
            per_head=self.config.per_head_scale,
        )
        v_quantized, v_quant_meta = quantize_fn(
            v_transformed,
            bits=self.config.bits_per_element,
            per_head=self.config.per_head_scale,
        )

        # Pack
        self.compressed_k = pack_fn(k_quantized, self.config.bits_per_element)
        self.compressed_v = pack_fn(v_quantized, self.config.bits_per_element)

        # 메타데이터 저장
        self.k_metadata = {
            "transform": k_transform_meta,
            "quantization": k_quant_meta,
            "shape": tuple(k_tensor.shape),
        }
        self.v_metadata = {
            "transform": v_transform_meta,
            "quantization": v_quant_meta,
            "shape": tuple(v_tensor.shape),
        }

        self.is_compressed = True

    def decompress(
        self,
        unpack_fn,
        dequantize_fn,
        inverse_transform_fn,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        압축된 K, V를 복원

        Returns:
            (k_tensor, v_tensor)
        """
        if not self.is_compressed:
            raise RuntimeError("Page is not compressed")

        # Unpack
        k_quantized = unpack_fn(
            self.compressed_k,
            self.config.bits_per_element,
            self.k_metadata["shape"],
        )
        v_quantized = unpack_fn(
            self.compressed_v,
            self.config.bits_per_element,
            self.v_metadata["shape"],
        )

        # Dequantize
        k_transformed = dequantize_fn(
            k_quantized,
            self.k_metadata["quantization"],
        )
        v_transformed = dequantize_fn(
            v_quantized,
            self.v_metadata["quantization"],
        )

        # Inverse transform
        k_tensor = inverse_transform_fn(
            k_transformed,
            self.k_metadata["transform"],
        )
        v_tensor = inverse_transform_fn(
            v_transformed,
            self.v_metadata["transform"],
        )

        return k_tensor, v_tensor

    def get_compressed_size(self) -> int:
        """압축된 데이터 크기 (bytes)"""
        if not self.is_compressed:
            return 0

        k_size = self.compressed_k.numel() * self.compressed_k.element_size()
        v_size = self.compressed_v.numel() * self.compressed_v.element_size()

        # 메타데이터 크기 추정 (scale, zero-point)
        if self.config.per_head_scale:
            num_heads = self.k_metadata["shape"][0]
            metadata_size = num_heads * 2 * 4 * 2  # 4 bytes (fp32) * 2 (scale/zero) * 2 (K/V)
        else:
            metadata_size = 2 * 4 * 2  # 단일 scale/zero

        return k_size + v_size + metadata_size

    def get_original_size(self) -> int:
        """원본 데이터 크기 (bytes)"""
        if self.k_metadata is None:
            return 0

        k_shape = self.k_metadata["shape"]
        # 각 원소는 float16/bfloat16 (2 bytes)
        return torch.prod(torch.tensor(k_shape)).item() * 2 * 2  # K + V

    def get_compression_ratio(self) -> float:
        """압축비"""
        original = self.get_original_size()
        compressed = self.get_compressed_size()

        if compressed == 0:
            return 0.0

        return original / compressed

    def __repr__(self) -> str:
        status = "compressed" if self.is_compressed else "uncompressed"
        hot = " [HOT]" if self.is_hot else ""
        return (
            f"KVPage(L{self.layer_idx}, P{self.page_idx}, "
            f"tokens=[{self.token_start}:{self.token_end}], "
            f"{status}{hot})"
        )
