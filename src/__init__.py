"""
LLM KV Cache Compression System (LLM-zram)
"""

__version__ = "0.1.0"

# Make submodules easily accessible
from . import models
from . import compression
from . import cache
from . import analysis
from . import evaluation

__all__ = ["models", "compression", "cache", "analysis", "evaluation"]
