"""
SmolLM3 모델 로더 및 관리
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmolLMLoader:
    """SmolLM3 모델을 로드하고 관리하는 클래스"""

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace 모델 이름
            device: 디바이스 (cuda/cpu)
            dtype: 데이터 타입
            use_flash_attention: Flash Attention 사용 여부
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention

        self.model = None
        self.tokenizer = None
        self.config = None

    def load(self) -> None:
        """모델과 토크나이저를 로드"""
        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Device: {self.device}, dtype: {self.dtype}")

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드 설정
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": self.device if self.device != "cpu" else None,
        }

        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.config = self.model.config

        logger.info(f"Model loaded successfully")
        logger.info(f"Model config: {self.config}")

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        if self.config is None:
            return {}

        info = {
            "model_name": self.model_name,
            "num_layers": self.config.num_hidden_layers,
            "hidden_size": self.config.hidden_size,
            "num_attention_heads": self.config.num_attention_heads,
            "num_key_value_heads": getattr(self.config, "num_key_value_heads", self.config.num_attention_heads),
            "max_position_embeddings": self.config.max_position_embeddings,
            "vocab_size": self.config.vocab_size,
            "head_dim": self.config.hidden_size // self.config.num_attention_heads,
        }

        return info

    def get_kv_cache_shape(self, batch_size: int, seq_length: int) -> tuple:
        """
        KV 캐시의 예상 shape 반환

        Returns:
            (num_layers, batch_size, num_kv_heads, seq_length, head_dim)
        """
        info = self.get_model_info()
        return (
            info["num_layers"],
            batch_size,
            info["num_key_value_heads"],
            seq_length,
            info["head_dim"],
        )

    def estimate_kv_memory(self, seq_length: int, batch_size: int = 1) -> Dict[str, float]:
        """
        KV 캐시의 예상 메모리 사용량 계산 (bytes)

        Returns:
            딕셔너리: 각 dtype별 메모리 사용량
        """
        info = self.get_model_info()

        # (num_layers, batch_size, num_kv_heads, seq_length, head_dim)
        # K와 V 각각
        num_elements = (
            info["num_layers"] *
            batch_size *
            info["num_key_value_heads"] *
            seq_length *
            info["head_dim"] *
            2  # K와 V
        )

        return {
            "fp32": num_elements * 4,
            "fp16": num_elements * 2,
            "bf16": num_elements * 2,
            "int8": num_elements * 1,
            "int4": num_elements * 0.5,
            "num_elements": num_elements,
        }

    def format_memory(self, bytes_size: float) -> str:
        """메모리 크기를 human-readable 형식으로 변환"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} TB"
