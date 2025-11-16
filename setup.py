from setuptools import setup, find_packages

setup(
    name="llm-kv-zram",
    version="0.1.0",
    description="LLM용 zram: Long-context LLM을 위한 (Near-)Lossless KV 캐시 압축 레이어",
    author="LCL Research Team",
    author_email="",
    url="https://github.com/MisileLab/LCL",
    packages=[
        "llm_kv_zram",
        "llm_kv_zram.models",
        "llm_kv_zram.compression",
        "llm_kv_zram.compression.level1",
        "llm_kv_zram.compression.level2",
        "llm_kv_zram.cache",
        "llm_kv_zram.analysis",
        "llm_kv_zram.evaluation",
    ],
    package_dir={"llm_kv_zram": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.26.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "analysis": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "pandas>=2.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
