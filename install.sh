#!/bin/bash
# LLM-KV-ZRAM 설치 스크립트

set -e

echo "=== LLM-KV-ZRAM 패키지 설치 ==="

# 현재 디렉토리 확인
if [ ! -f "setup.py" ]; then
    echo "오류: setup.py를 찾을 수 없습니다."
    echo "이 스크립트는 프로젝트 루트 디렉토리에서 실행해야 합니다."
    exit 1
fi

# uv 사용 가능 여부 확인
if command -v uv &> /dev/null; then
    echo "uv를 사용하여 설치합니다..."
    uv pip install -e .
elif command -v pip &> /dev/null; then
    echo "pip을 사용하여 설치합니다..."
    pip install -e .
else
    echo "오류: pip 또는 uv를 찾을 수 없습니다."
    exit 1
fi

echo ""
echo "✓ 설치 완료!"
echo ""
echo "테스트하려면 다음 명령어를 실행하세요:"
echo "  python -c 'from models import SmolLMLoader; print(\"Import successful!\")'"
echo ""
echo "스크립트 실행:"
echo "  python scripts/analyze_kv.py --help"
echo "  python experiments/m1_baseline.py --help"
