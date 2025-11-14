"""
Level 2 압축 - 엔트로피 코딩 (CPU/오프라인)
- 페이지 단위 Huffman/ANS 계열 엔트로피 코딩
"""

import torch
import numpy as np
from collections import Counter
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EntropyCoder:
    """
    페이지 단위 엔트로피 코더
    - CPU 사이드에서 동작
    - Level 1 압축 후 추가 압축
    """

    def __init__(self):
        pass

    def encode(
        self,
        data: torch.Tensor,
        method: str = "huffman",
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        데이터를 엔트로피 코딩으로 압축

        Args:
            data: 압축할 데이터 (uint8 or int32)
            method: 압축 방법 ("huffman", "rle", "simple")

        Returns:
            (compressed_bytes, metadata)
        """
        if method == "huffman":
            return self._encode_huffman(data)
        elif method == "rle":
            return self._encode_rle(data)
        elif method == "simple":
            # 간단한 압축 (delta encoding + RLE)
            return self._encode_simple(data)
        else:
            raise ValueError(f"Unknown method: {method}")

    def decode(
        self,
        compressed: bytes,
        metadata: Dict[str, Any],
    ) -> torch.Tensor:
        """
        압축된 데이터를 복원

        Args:
            compressed: 압축된 바이트
            metadata: 압축 시 저장된 메타데이터

        Returns:
            복원된 데이터
        """
        method = metadata["method"]

        if method == "huffman":
            return self._decode_huffman(compressed, metadata)
        elif method == "rle":
            return self._decode_rle(compressed, metadata)
        elif method == "simple":
            return self._decode_simple(compressed, metadata)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _encode_huffman(
        self,
        data: torch.Tensor,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        간단한 Huffman 스타일 압축
        (실제로는 단순화된 버전)

        Note: 실제 Huffman 구현은 복잡하므로, 여기서는
        빈도 기반 delta encoding으로 근사
        """
        # CPU로 이동
        data_cpu = data.cpu().numpy().flatten()

        # 빈도 계산
        unique, counts = np.unique(data_cpu, return_counts=True)
        freq = dict(zip(unique.tolist(), counts.tolist()))

        # 간단한 압축: delta encoding
        if len(data_cpu) > 1:
            first_val = data_cpu[0]
            deltas = np.diff(data_cpu)

            # Delta를 bytes로 변환 (int8로 가정)
            deltas_bytes = deltas.astype(np.int8).tobytes()
        else:
            first_val = data_cpu[0]
            deltas_bytes = b""

        metadata = {
            "method": "huffman",
            "shape": tuple(data.shape),
            "dtype": str(data.dtype),
            "first_val": int(first_val),
            "length": len(data_cpu),
            "freq": freq,
        }

        compressed = deltas_bytes

        return compressed, metadata

    def _decode_huffman(
        self,
        compressed: bytes,
        metadata: Dict[str, Any],
    ) -> torch.Tensor:
        """Huffman 복원"""
        first_val = metadata["first_val"]
        length = metadata["length"]
        shape = metadata["shape"]

        if length == 1:
            data = np.array([first_val])
        else:
            # Delta 복원
            deltas = np.frombuffer(compressed, dtype=np.int8)

            # Cumsum으로 원본 복원
            data = np.zeros(length, dtype=np.int32)
            data[0] = first_val
            data[1:] = first_val + np.cumsum(deltas)

        # Tensor로 변환
        tensor = torch.from_numpy(data).reshape(shape)

        return tensor

    def _encode_rle(
        self,
        data: torch.Tensor,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Run-Length Encoding

        연속된 같은 값을 (값, 길이) 쌍으로 압축
        """
        data_cpu = data.cpu().numpy().flatten()

        if len(data_cpu) == 0:
            return b"", {"method": "rle", "shape": tuple(data.shape), "dtype": str(data.dtype)}

        # RLE 인코딩
        rle_pairs = []
        current_val = data_cpu[0]
        current_count = 1

        for val in data_cpu[1:]:
            if val == current_val:
                current_count += 1
            else:
                rle_pairs.append((current_val, current_count))
                current_val = val
                current_count = 1

        rle_pairs.append((current_val, current_count))

        # Bytes로 변환 (값: int32, 길이: int32)
        rle_bytes = b""
        for val, count in rle_pairs:
            rle_bytes += int(val).to_bytes(4, byteorder="little", signed=True)
            rle_bytes += int(count).to_bytes(4, byteorder="little", signed=False)

        metadata = {
            "method": "rle",
            "shape": tuple(data.shape),
            "dtype": str(data.dtype),
            "length": len(data_cpu),
        }

        return rle_bytes, metadata

    def _decode_rle(
        self,
        compressed: bytes,
        metadata: Dict[str, Any],
    ) -> torch.Tensor:
        """RLE 복원"""
        shape = metadata["shape"]
        length = metadata["length"]

        # RLE 페어 파싱
        data = []
        i = 0
        while i < len(compressed):
            val = int.from_bytes(compressed[i:i+4], byteorder="little", signed=True)
            count = int.from_bytes(compressed[i+4:i+8], byteorder="little", signed=False)
            data.extend([val] * count)
            i += 8

        data = np.array(data[:length], dtype=np.int32)
        tensor = torch.from_numpy(data).reshape(shape)

        return tensor

    def _encode_simple(
        self,
        data: torch.Tensor,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        간단한 압축 (placeholder)
        - 실제로는 zlib 등을 사용할 수 있음
        """
        import zlib

        data_bytes = data.cpu().numpy().tobytes()
        compressed = zlib.compress(data_bytes, level=9)

        metadata = {
            "method": "simple",
            "shape": tuple(data.shape),
            "dtype": str(data.dtype),
        }

        return compressed, metadata

    def _decode_simple(
        self,
        compressed: bytes,
        metadata: Dict[str, Any],
    ) -> torch.Tensor:
        """간단한 압축 복원"""
        import zlib

        shape = metadata["shape"]
        dtype_str = metadata["dtype"]

        # dtype 파싱
        if "int32" in dtype_str:
            dtype = np.int32
        elif "int8" in dtype_str:
            dtype = np.int8
        elif "uint8" in dtype_str:
            dtype = np.uint8
        else:
            dtype = np.int32

        decompressed = zlib.decompress(compressed)
        data = np.frombuffer(decompressed, dtype=dtype)
        tensor = torch.from_numpy(data).reshape(shape)

        return tensor

    def estimate_compression_ratio(
        self,
        data: torch.Tensor,
        method: str = "huffman",
    ) -> float:
        """
        압축비 추정

        Returns:
            원본 크기 / 압축 크기
        """
        original_size = data.numel() * data.element_size()

        compressed, metadata = self.encode(data, method)
        compressed_size = len(compressed)

        if compressed_size == 0:
            return 0.0

        return original_size / compressed_size
