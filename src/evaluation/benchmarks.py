"""
벤치마크 스위트
- Long-context 벤치마크
- QA 태스크
"""

import torch
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class PasskeyRetrieval:
    """
    Passkey Retrieval 벤치마크
    - Long-context에서 특정 정보(passkey) 추출 능력 평가
    """

    def __init__(self, context_length: int = 8192):
        self.context_length = context_length

    def generate_sample(
        self,
        passkey: str = "12345",
        position_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Passkey retrieval 샘플 생성

        Args:
            passkey: 숨길 정보
            position_ratio: 위치 비율 (0~1)

        Returns:
            샘플 딕셔너리
        """
        # 간단한 filler 텍스트
        filler = "This is a random sentence. " * 50

        # Passkey 문장
        passkey_sentence = f"The secret passkey is {passkey}. Remember this."

        # 위치 계산
        filler_tokens_before = int(self.context_length * position_ratio) // 10
        filler_tokens_after = (self.context_length - filler_tokens_before - 100) // 10

        # 컨텍스트 구성
        context = (
            filler * filler_tokens_before +
            passkey_sentence +
            filler * filler_tokens_after
        )

        # 질문
        question = "What is the secret passkey?"

        return {
            "context": context,
            "question": question,
            "answer": passkey,
            "position_ratio": position_ratio,
        }

    def evaluate(
        self,
        model,
        tokenizer,
        num_samples: int = 10,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """
        Passkey retrieval 평가

        Returns:
            정확도 딕셔너리
        """
        correct = 0
        total = num_samples

        for i in range(num_samples):
            # 샘플 생성
            position_ratio = (i + 1) / (num_samples + 1)
            sample = self.generate_sample(
                passkey=f"{i:05d}",
                position_ratio=position_ratio,
            )

            # 프롬프트
            prompt = sample["context"] + "\n\n" + sample["question"] + "\nAnswer:"

            # 생성
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=20,
                    do_sample=False,
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 답변 추출 (간단히 마지막 라인)
            answer = generated.split("Answer:")[-1].strip()

            # 정답 확인
            if sample["answer"] in answer:
                correct += 1

            logger.info(f"Sample {i}: Expected={sample['answer']}, Got={answer}")

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }


class SimpleQA:
    """간단한 QA 벤치마크"""

    def __init__(self):
        # 간단한 QA 샘플들
        self.samples = [
            {
                "context": "Paris is the capital of France. It is known for the Eiffel Tower.",
                "question": "What is the capital of France?",
                "answer": "Paris",
            },
            {
                "context": "The Earth orbits the Sun. It takes approximately 365 days for one orbit.",
                "question": "How long does it take for Earth to orbit the Sun?",
                "answer": "365 days",
            },
            {
                "context": "Python is a high-level programming language. It was created by Guido van Rossum.",
                "question": "Who created Python?",
                "answer": "Guido van Rossum",
            },
        ]

    def evaluate(
        self,
        model,
        tokenizer,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """
        QA 평가

        Returns:
            정확도 딕셔너리
        """
        correct = 0
        total = len(self.samples)

        for i, sample in enumerate(self.samples):
            # 프롬프트
            prompt = (
                f"Context: {sample['context']}\n\n"
                f"Question: {sample['question']}\n"
                f"Answer:"
            )

            # 생성
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=30,
                    do_sample=False,
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 답변 추출
            answer = generated.split("Answer:")[-1].strip()

            # 정답 확인 (간단히 포함 여부)
            if sample["answer"].lower() in answer.lower():
                correct += 1

            logger.info(f"Sample {i}: Expected={sample['answer']}, Got={answer}")

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
