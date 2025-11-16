#!/bin/bash
# 가상환경과 함께 LCL 스크립트 실행

# 스크립트가 있는 디렉토리
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_DIR="$SCRIPT_DIR/src"

# PYTHONPATH 설정
export PYTHONPATH="$SRC_DIR:$PYTHONPATH"

echo "========================================="
echo "LCL Project Environment"
echo "========================================="
echo "Project root: $SCRIPT_DIR"
echo "PYTHONPATH:   $PYTHONPATH"
echo ""

# 가상환경 확인
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✓ Virtual environment detected:"
    echo "  $VIRTUAL_ENV"
    echo "  Python: $(which python)"
else
    echo "⚠ No virtual environment detected"
    echo "  Using system Python: $(which python)"
fi

echo "========================================="
echo ""

# 인자가 있으면 그대로 실행
if [ $# -gt 0 ]; then
    echo "Running: $@"
    echo ""
    exec "$@"
else
    echo "Usage: $0 <command>"
    echo ""
    echo "Examples:"
    echo "  $0 python scripts/analyze_kv.py --context-length 1024"
    echo "  $0 python scripts/run_baseline.py --context-length 1024"
fi
