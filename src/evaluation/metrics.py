"""
평가 메트릭
- 성능: tokens/s, 메모리, latency
- 품질: perplexity, accuracy
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
import time
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """성능 메트릭 수집 및 계산"""

    @staticmethod
    def measure_memory_usage(device: str = "cuda") -> Dict[str, int]:
        """
        GPU 메모리 사용량 측정

        Returns:
            딕셔너리: allocated, reserved, max_allocated (bytes)
        """
        if device == "cuda" and torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated(),
            }
        else:
            return {"allocated": 0, "reserved": 0, "max_allocated": 0}

    @staticmethod
    def format_memory(bytes_size: int) -> str:
        """메모리 크기 포맷"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} TB"

    @staticmethod
    def calculate_tokens_per_sec(num_tokens: int, elapsed_time: float) -> float:
        """Tokens/s 계산"""
        if elapsed_time == 0:
            return 0.0
        return num_tokens / elapsed_time

    @staticmethod
    def measure_latency(
        fn,
        *args,
        device: str = "cuda",
        **kwargs
    ) -> tuple:
        """
        함수 실행 시간 측정

        Returns:
            (result, elapsed_time)
        """
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = fn(*args, **kwargs)

        if device == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        return result, end - start


class QualityMetrics:
    """품질 메트릭 (Perplexity, Accuracy 등)"""

    @staticmethod
    @torch.no_grad()
    def calculate_perplexity(
        model,
        tokenizer,
        text: str,
        max_length: int = 512,
        device: str = "cuda",
    ) -> float:
        """
        Perplexity 계산

        Args:
            model: 언어 모델
            tokenizer: 토크나이저
            text: 평가 텍스트
            max_length: 최대 길이
            device: 디바이스

        Returns:
            perplexity 값
        """
        # 토큰화
        encodings = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings.get("attention_mask", None)

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        # Loss (negative log-likelihood)
        loss = outputs.loss

        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()

        return perplexity

    @staticmethod
    @torch.no_grad()
    def calculate_perplexity_kv(
        model,
        tokenizer,
        text: str,
        past_key_values: Optional[tuple] = None,
        max_length: int = 512,
        device: str = "cuda",
    ) -> float:
        """
        KV 캐시를 사용한 Perplexity 계산

        Args:
            model: 언어 모델
            tokenizer: 토크나이저
            text: 평가 텍스트
            past_key_values: 기존 KV 캐시 (옵션)
            max_length: 최대 길이
            device: 디바이스

        Returns:
            perplexity 값
        """
        # 토큰화
        encodings = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )

        input_ids = encodings["input_ids"].to(device)

        # Forward pass with KV cache
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits = outputs.logits

        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Calculate cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )

        # Perplexity
        perplexity = torch.exp(loss).item()

        return perplexity

    @staticmethod
    def compare_outputs(
        output1: str,
        output2: str,
    ) -> Dict[str, float]:
        """
        두 출력 비교 (문자열 유사도)

        Returns:
            similarity 메트릭
        """
        # 간단한 문자열 유사도
        len1 = len(output1)
        len2 = len(output2)

        if len1 == 0 and len2 == 0:
            return {"exact_match": 1.0, "length_ratio": 1.0}

        exact_match = 1.0 if output1 == output2 else 0.0
        length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0

        # Character-level similarity (간단한 버전)
        common_chars = sum(1 for c1, c2 in zip(output1, output2) if c1 == c2)
        char_similarity = common_chars / max(len1, len2) if max(len1, len2) > 0 else 0.0

        return {
            "exact_match": exact_match,
            "length_ratio": length_ratio,
            "char_similarity": char_similarity,
        }

    @staticmethod
    @torch.no_grad()
    def calculate_kl_divergence(
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        temperature: float = 1.0,
    ) -> float:
        """
        두 로짓 분포 간 KL divergence 계산

        Args:
            logits1: (batch, seq_len, vocab_size)
            logits2: (batch, seq_len, vocab_size)
            temperature: softmax temperature

        Returns:
            KL divergence 값
        """
        # Softmax with temperature
        p = F.softmax(logits1 / temperature, dim=-1)
        q = F.softmax(logits2 / temperature, dim=-1)

        # KL(P || Q) = Σ P(x) log(P(x) / Q(x))
        kl = F.kl_div(
            q.log(),
            p,
            reduction="batchmean",
            log_target=False,
        )

        return kl.item()
