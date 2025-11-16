#!/bin/bash
# Test script to verify setup_env.sh works

echo "Testing setup_env.sh..."
source setup_env.sh

echo ""
echo "Checking if modules can be imported..."
python3 -c "
import sys
print('PYTHONPATH from env:', sys.path[0] if sys.path else 'None')

try:
    from models import SmolLMLoader
    print('✓ models.SmolLMLoader imported')
except ImportError as e:
    print('✗ Failed to import models:', e)
    
try:
    from analysis import KVAnalyzer  
    print('✓ analysis.KVAnalyzer imported')
except ImportError as e:
    print('✗ Failed to import analysis:', e)
"
