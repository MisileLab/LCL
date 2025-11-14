"""
캐시 정책 (Hot/Cold 페이지 관리)
"""

from collections import OrderedDict
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CachePolicy:
    """캐시 정책 베이스 클래스"""

    def access(self, layer_idx: int, page_idx: int):
        """페이지 접근 기록"""
        raise NotImplementedError

    def should_evict(self) -> bool:
        """Eviction이 필요한지 확인"""
        raise NotImplementedError

    def evict(self) -> Tuple[int, int]:
        """
        Evict할 페이지 선택

        Returns:
            (layer_idx, page_idx)
        """
        raise NotImplementedError


class LRUPolicy(CachePolicy):
    """LRU (Least Recently Used) 정책"""

    def __init__(self, capacity: int = 32):
        """
        Args:
            capacity: 최대 Hot 페이지 수
        """
        self.capacity = capacity
        # OrderedDict: (layer_idx, page_idx) -> access count
        self.cache = OrderedDict()

    def access(self, layer_idx: int, page_idx: int):
        """페이지 접근 기록"""
        key = (layer_idx, page_idx)

        # 이미 있으면 맨 뒤로 이동 (최근 사용)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = True

    def should_evict(self) -> bool:
        """Eviction 필요 여부"""
        return len(self.cache) >= self.capacity

    def evict(self) -> Tuple[int, int]:
        """가장 오래 사용되지 않은 페이지 제거"""
        if not self.cache:
            raise RuntimeError("No pages to evict")

        # 맨 앞 (가장 오래된) 항목
        key, _ = self.cache.popitem(last=False)

        return key


class FIFOPolicy(CachePolicy):
    """FIFO (First In First Out) 정책"""

    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self.queue = []

    def access(self, layer_idx: int, page_idx: int):
        """페이지 접근 기록"""
        key = (layer_idx, page_idx)

        if key not in self.queue:
            self.queue.append(key)

    def should_evict(self) -> bool:
        return len(self.queue) >= self.capacity

    def evict(self) -> Tuple[int, int]:
        """가장 먼저 들어온 페이지 제거"""
        if not self.queue:
            raise RuntimeError("No pages to evict")

        return self.queue.pop(0)
