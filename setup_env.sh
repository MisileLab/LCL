#!/bin/bash
# LCL 프로젝트 환경 설정

# 스크립트가 있는 디렉토리를 찾음
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_DIR="$SCRIPT_DIR/src"

# PYTHONPATH에 src 디렉토리 추가
export PYTHONPATH="$SRC_DIR:$PYTHONPATH"

echo "✓ PYTHONPATH 설정 완료"
echo "  Project: $SCRIPT_DIR"
echo "  Src:     $SRC_DIR"
echo ""
echo "이제 스크립트를 실행할 수 있습니다:"
echo "  python scripts/analyze_kv.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --context-length 1024"
echo "  python scripts/run_baseline.py --context-length 1024"
