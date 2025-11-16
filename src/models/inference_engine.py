"""
커스텀 Inference 엔진 (Prefill + Decode with KV cache management)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """추론 성능 메트릭"""
    prefill_time: float = 0.0
    decode_time: float = 0.0
    total_time: float = 0.0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_tokens_per_sec: float = 0.0
    decode_tokens_per_sec: float = 0.0
    kv_memory_bytes: int = 0
    peak_memory_bytes: int = 0


class InferenceEngine:
    """
    커스텀 Inference 엔진
    - Prefill: 입력 시퀀스를 한 번에 처리
    - Decode: 한 토큰씩 생성
    - KV 캐시 관리 (나중에 압축 레이어 통합 예정)
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        use_cache: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_cache = use_cache

        # KV 캐시 저장소 (나중에 압축 매니저로 교체)
        self.kv_cache: Optional[Tuple] = None

    def prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """입력 준비"""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Prefill 단계: 입력 시퀀스를 한 번에 처리하고 KV 캐시 생성

        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)

        Returns:
            logits: (batch_size, seq_length, vocab_size)
            past_key_values: KV 캐시
        """
        inputs = self.prepare_inputs(input_ids, attention_mask)

        outputs = self.model(
            **inputs,
            use_cache=self.use_cache,
            return_dict=True,
        )

        return outputs.logits, outputs.past_key_values

    @torch.no_grad()
    def decode_step(
        self,
        token_id: torch.Tensor,
        past_key_values: Tuple,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Decode 단계: 한 토큰씩 생성

        Args:
            token_id: (batch_size, 1) - 현재 토큰
            past_key_values: KV 캐시
            attention_mask: (batch_size, current_length)

        Returns:
            logits: (batch_size, 1, vocab_size)
            updated_past_key_values: 업데이트된 KV 캐시
        """
        outputs = self.model(
            input_ids=token_id.to(self.device),
            attention_mask=attention_mask.to(self.device),
            past_key_values=past_key_values,
            use_cache=self.use_cache,
            return_dict=True,
        )

        return outputs.logits, outputs.past_key_values

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        do_sample: bool = True,
        collect_metrics: bool = True,
    ) -> Tuple[str, Optional[InferenceMetrics]]:
        """
        텍스트 생성 (Prefill + Decode)

        Args:
            prompt: 입력 프롬프트
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 온도
            top_k: top-k 샘플링
            top_p: nucleus 샘플링
            do_sample: 샘플링 여부
            collect_metrics: 메트릭 수집 여부

        Returns:
            생성된 텍스트, 메트릭 (옵션)
        """
        # 입력 토큰화
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        batch_size, prefill_length = input_ids.shape

        metrics = InferenceMetrics() if collect_metrics else None

        # Memory tracking
        if collect_metrics and self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Prefill 단계
        prefill_start = time.perf_counter()

        logits, past_key_values = self.prefill(input_ids, attention_mask)

        if self.device == "cuda":
            torch.cuda.synchronize()
        prefill_end = time.perf_counter()

        if collect_metrics:
            metrics.prefill_time = prefill_end - prefill_start
            metrics.prefill_tokens = prefill_length
            metrics.prefill_tokens_per_sec = prefill_length / metrics.prefill_time

        # 첫 번째 토큰 샘플링
        next_token_logits = logits[:, -1, :] / temperature
        next_token_id = self._sample_token(next_token_logits, top_k, top_p, do_sample)

        # 생성된 토큰들
        generated_ids = [next_token_id.item()]

        # Decode 단계
        decode_start = time.perf_counter()

        for _ in range(max_new_tokens - 1):
            # attention mask 업데이트
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=self.device)
            ], dim=1)

            # Decode step
            logits, past_key_values = self.decode_step(
                next_token_id.unsqueeze(1),
                past_key_values,
                attention_mask,
            )

            # 다음 토큰 샘플링
            next_token_logits = logits[:, -1, :] / temperature
            next_token_id = self._sample_token(next_token_logits, top_k, top_p, do_sample)

            generated_ids.append(next_token_id.item())

            # EOS 체크
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        if self.device == "cuda":
            torch.cuda.synchronize()
        decode_end = time.perf_counter()

        if collect_metrics:
            metrics.decode_time = decode_end - decode_start
            metrics.decode_tokens = len(generated_ids)
            metrics.total_time = metrics.prefill_time + metrics.decode_time
            if metrics.decode_time > 0:
                metrics.decode_tokens_per_sec = metrics.decode_tokens / metrics.decode_time

            # KV 캐시 메모리 계산
            if past_key_values is not None:
                kv_memory = 0
                for layer_kv in past_key_values:
                    for kv in layer_kv:
                        if kv is not None:
                            kv_memory += kv.numel() * kv.element_size()
                metrics.kv_memory_bytes = kv_memory

            if self.device == "cuda":
                metrics.peak_memory_bytes = torch.cuda.max_memory_allocated()

        # 디코딩
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = prompt + generated_text

        return full_text, metrics

    def _sample_token(
        self,
        logits: torch.Tensor,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
    ) -> torch.Tensor:
        """토큰 샘플링"""
        if not do_sample:
            return torch.argmax(logits, dim=-1)

        # Top-k 필터링
        if top_k is not None and top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) 필터링
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 누적 확률이 top_p를 초과하는 토큰 제거
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # 샘플링
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token.squeeze(-1)

    def get_kv_cache_stats(self, past_key_values: Tuple) -> Dict[str, Any]:
        """KV 캐시 통계 반환"""
        if past_key_values is None:
            return {}

        num_layers = len(past_key_values)

        # 첫 번째 레이어의 K, V shape
        first_k = past_key_values[0][0]
        first_v = past_key_values[0][1]

        total_memory = 0
        for layer_kv in past_key_values:
            for kv in layer_kv:
                if kv is not None:
                    total_memory += kv.numel() * kv.element_size()

        return {
            "num_layers": num_layers,
            "k_shape": tuple(first_k.shape) if first_k is not None else None,
            "v_shape": tuple(first_v.shape) if first_v is not None else None,
            "total_memory_bytes": total_memory,
            "dtype": str(first_k.dtype) if first_k is not None else None,
        }
