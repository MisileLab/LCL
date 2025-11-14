"""
KV 캐시 dump 및 로드 도구
"""

import torch
import pickle
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class KVDumper:
    """KV 캐시를 파일로 저장하고 로드하는 도구"""

    def __init__(self, output_dir: str = "./kv_dumps"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def dump_kv_cache(
        self,
        past_key_values: Tuple,
        prefix: str = "kv_cache",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        KV 캐시를 파일로 저장

        Args:
            past_key_values: 저장할 KV 캐시
            prefix: 파일명 prefix
            metadata: 추가 메타데이터

        Returns:
            저장된 파일 경로
        """
        # KV 캐시를 CPU로 이동 (저장용)
        kv_cpu = []
        for k, v in past_key_values:
            kv_cpu.append((k.cpu(), v.cpu()))

        # 메타데이터 구성
        if metadata is None:
            metadata = {}

        metadata.update({
            "num_layers": len(kv_cpu),
            "k_shape": tuple(kv_cpu[0][0].shape),
            "v_shape": tuple(kv_cpu[0][1].shape),
            "dtype": str(kv_cpu[0][0].dtype),
        })

        # 저장
        save_path = self.output_dir / f"{prefix}.pt"
        metadata_path = self.output_dir / f"{prefix}_metadata.json"

        torch.save(kv_cpu, save_path)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"KV cache dumped to: {save_path}")
        logger.info(f"Metadata saved to: {metadata_path}")

        return str(save_path)

    def load_kv_cache(
        self,
        file_path: str,
        device: str = "cuda",
    ) -> Tuple[Tuple, Dict[str, Any]]:
        """
        KV 캐시를 파일에서 로드

        Args:
            file_path: 로드할 파일 경로
            device: 로드할 디바이스

        Returns:
            (past_key_values, metadata)
        """
        file_path = Path(file_path)
        metadata_path = file_path.parent / f"{file_path.stem}_metadata.json"

        # KV 캐시 로드
        kv_cache = torch.load(file_path, map_location=device)

        # 메타데이터 로드
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        logger.info(f"KV cache loaded from: {file_path}")

        return tuple(kv_cache), metadata

    def dump_layer_wise(
        self,
        past_key_values: Tuple,
        prefix: str = "kv_layer",
    ) -> list:
        """
        레이어별로 KV 캐시를 개별 파일로 저장

        Returns:
            저장된 파일 경로 리스트
        """
        saved_paths = []

        for layer_idx, (k, v) in enumerate(past_key_values):
            layer_path = self.output_dir / f"{prefix}_L{layer_idx:02d}.pt"

            torch.save({
                "k": k.cpu(),
                "v": v.cpu(),
                "layer_idx": layer_idx,
                "k_shape": tuple(k.shape),
                "v_shape": tuple(v.shape),
            }, layer_path)

            saved_paths.append(str(layer_path))

        logger.info(f"Dumped {len(saved_paths)} layers to: {self.output_dir}")

        return saved_paths

    def load_layer_wise(
        self,
        prefix: str = "kv_layer",
        device: str = "cuda",
    ) -> Tuple:
        """
        레이어별로 저장된 KV 캐시를 로드

        Returns:
            past_key_values (tuple of tuples)
        """
        # 파일 찾기
        layer_files = sorted(self.output_dir.glob(f"{prefix}_L*.pt"))

        if not layer_files:
            raise FileNotFoundError(f"No layer files found with prefix: {prefix}")

        kv_cache = []

        for layer_file in layer_files:
            layer_data = torch.load(layer_file, map_location=device)
            kv_cache.append((layer_data["k"], layer_data["v"]))

        logger.info(f"Loaded {len(kv_cache)} layers from: {self.output_dir}")

        return tuple(kv_cache)

    def dump_for_analysis(
        self,
        past_key_values: Tuple,
        analysis_results: Dict[str, Any],
        prefix: str = "analysis",
    ) -> str:
        """
        분석 결과와 함께 KV 캐시 저장

        Returns:
            저장된 디렉토리 경로
        """
        analysis_dir = self.output_dir / prefix
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # KV 캐시 저장
        kv_path = analysis_dir / "kv_cache.pt"
        kv_cpu = [(k.cpu(), v.cpu()) for k, v in past_key_values]
        torch.save(kv_cpu, kv_path)

        # 분석 결과 저장
        analysis_path = analysis_dir / "analysis_results.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis_results, f, indent=2)

        logger.info(f"Analysis dump saved to: {analysis_dir}")

        return str(analysis_dir)
