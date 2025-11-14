"""
KV 페이지 관리자
- Hot/Cold 페이지 정책 관리
- 압축/복원 조율
"""

import torch
from typing import Dict, List, Optional, Tuple
from ..compression.kv_page import KVPage, KVPageConfig
from ..compression.level1.transform import KVTransform
from ..compression.level1.quantization import NearLosslessQuantizer
from ..compression.level1.bit_packing import BitPacker
from .policies import CachePolicy, LRUPolicy
import logging

logger = logging.getLogger(__name__)


class PageManager:
    """
    KV 캐시 페이지 관리자
    - 페이지 단위로 압축/복원 관리
    - Hot/Cold 정책 적용
    """

    def __init__(
        self,
        config: KVPageConfig,
        cache_policy: Optional[CachePolicy] = None,
        num_layers: int = 32,
    ):
        """
        Args:
            config: 페이지 설정
            cache_policy: 캐시 정책 (기본: LRU)
            num_layers: 레이어 수
        """
        self.config = config
        self.num_layers = num_layers

        # 캐시 정책
        if cache_policy is None:
            cache_policy = LRUPolicy(capacity=32)  # 기본 32 페이지
        self.cache_policy = cache_policy

        # 압축 컴포넌트
        self.transform = KVTransform(
            transform_type=config.transform_type,
            device=config.device,
        )
        self.quantizer = NearLosslessQuantizer(device=config.device)
        self.packer = BitPacker(device=config.device)

        # 페이지 저장소
        # pages[layer_idx][page_idx] = KVPage
        self.pages: Dict[int, Dict[int, KVPage]] = {}
        for layer_idx in range(num_layers):
            self.pages[layer_idx] = {}

        # Hot cache (복원된 상태로 메모리에 유지)
        # hot_cache[layer_idx][page_idx] = (k_tensor, v_tensor)
        self.hot_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
        for layer_idx in range(num_layers):
            self.hot_cache[layer_idx] = {}

    def add_kv(
        self,
        layer_idx: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        token_start: int,
        compress: bool = True,
    ):
        """
        KV 텐서를 페이지로 추가

        Args:
            layer_idx: 레이어 인덱스
            k_tensor: (num_heads, num_tokens, head_dim)
            v_tensor: (num_heads, num_tokens, head_dim)
            token_start: 시작 토큰 위치
            compress: 즉시 압축 여부
        """
        num_tokens = k_tensor.shape[1]
        page_size = self.config.page_size

        # 페이지로 분할
        for offset in range(0, num_tokens, page_size):
            token_end = min(offset + page_size, num_tokens)
            page_idx = (token_start + offset) // page_size

            # 페이지 생성
            page = KVPage(
                layer_idx=layer_idx,
                page_idx=page_idx,
                token_start=token_start + offset,
                token_end=token_start + token_end,
                config=self.config,
            )

            # K, V 슬라이스
            k_slice = k_tensor[:, offset:token_end, :]
            v_slice = v_tensor[:, offset:token_end, :]

            if compress:
                # 압축
                page.compress(
                    k_slice,
                    v_slice,
                    transform_fn=self.transform.transform,
                    quantize_fn=self.quantizer.quantize,
                    pack_fn=self.packer.pack,
                )
                self.pages[layer_idx][page_idx] = page
            else:
                # Hot cache에 직접 저장
                self.hot_cache[layer_idx][page_idx] = (k_slice, v_slice)

            # 캐시 정책 업데이트
            self.cache_policy.access(layer_idx, page_idx)

    def get_kv(
        self,
        layer_idx: int,
        page_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        페이지의 KV 텐서 가져오기 (자동 복원)

        Args:
            layer_idx: 레이어 인덱스
            page_idx: 페이지 인덱스

        Returns:
            (k_tensor, v_tensor)
        """
        # Hot cache에 있는지 확인
        if page_idx in self.hot_cache[layer_idx]:
            self.cache_policy.access(layer_idx, page_idx)
            return self.hot_cache[layer_idx][page_idx]

        # Cold 페이지에서 복원
        if page_idx not in self.pages[layer_idx]:
            raise KeyError(f"Page not found: layer={layer_idx}, page={page_idx}")

        page = self.pages[layer_idx][page_idx]

        # 복원
        k_tensor, v_tensor = page.decompress(
            unpack_fn=self.packer.unpack,
            dequantize_fn=self.quantizer.dequantize,
            inverse_transform_fn=self.transform.inverse_transform,
        )

        # Hot cache에 추가
        self._add_to_hot_cache(layer_idx, page_idx, k_tensor, v_tensor)

        # 캐시 정책 업데이트
        self.cache_policy.access(layer_idx, page_idx)

        return k_tensor, v_tensor

    def get_kv_range(
        self,
        layer_idx: int,
        token_start: int,
        token_end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        토큰 범위에 해당하는 KV 가져오기

        Args:
            layer_idx: 레이어 인덱스
            token_start: 시작 토큰
            token_end: 끝 토큰

        Returns:
            (k_tensor, v_tensor) - 연결된 텐서
        """
        page_size = self.config.page_size
        start_page = token_start // page_size
        end_page = (token_end - 1) // page_size

        k_tensors = []
        v_tensors = []

        for page_idx in range(start_page, end_page + 1):
            k, v = self.get_kv(layer_idx, page_idx)
            k_tensors.append(k)
            v_tensors.append(v)

        # 연결
        k_full = torch.cat(k_tensors, dim=1)
        v_full = torch.cat(v_tensors, dim=1)

        # 정확한 범위로 슬라이스
        start_offset = token_start % page_size
        end_offset = start_offset + (token_end - token_start)

        if len(k_tensors) == 1:
            k_full = k_full[:, start_offset:end_offset, :]
            v_full = v_full[:, start_offset:end_offset, :]

        return k_full, v_full

    def _add_to_hot_cache(
        self,
        layer_idx: int,
        page_idx: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
    ):
        """Hot cache에 추가 (정책에 따라 eviction)"""
        # Eviction 필요한지 확인
        if self.cache_policy.should_evict():
            evict_layer, evict_page = self.cache_policy.evict()
            if evict_page in self.hot_cache[evict_layer]:
                del self.hot_cache[evict_layer][evict_page]

        # 추가
        self.hot_cache[layer_idx][page_idx] = (k_tensor, v_tensor)

    def get_stats(self) -> Dict:
        """통계 반환"""
        total_pages = sum(len(pages) for pages in self.pages.values())
        total_hot = sum(len(cache) for cache in self.hot_cache.values())

        total_compressed_size = 0
        total_original_size = 0

        for layer_pages in self.pages.values():
            for page in layer_pages.values():
                total_compressed_size += page.get_compressed_size()
                total_original_size += page.get_original_size()

        avg_compression_ratio = 0.0
        if total_compressed_size > 0:
            avg_compression_ratio = total_original_size / total_compressed_size

        return {
            "total_pages": total_pages,
            "total_hot_pages": total_hot,
            "total_compressed_bytes": total_compressed_size,
            "total_original_bytes": total_original_size,
            "avg_compression_ratio": avg_compression_ratio,
        }
