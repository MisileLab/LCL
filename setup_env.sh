#!/bin/bash
# LCL 프로젝트 환경 설정

# PYTHONPATH에 src 디렉토리 추가
export PYTHONPATH="/home/user/LCL/src:$PYTHONPATH"

echo "PYTHONPATH가 설정되었습니다: $PYTHONPATH"
echo ""
echo "이제 스크립트를 실행할 수 있습니다:"
echo "  python scripts/analyze_kv.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --context-length 1024"
echo "  python scripts/run_baseline.py --context-length 1024"
