#!/usr/bin/env python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="purpose",
    version="0.1.0",
    author="Sprint Team",
    author_email="example@example.com",
    description="A framework for creating purpose-specific language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/purpose",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.25.0",
        "datasets>=2.8.0",
        "pandas>=1.3.0",
        "tqdm>=4.64.0",
        "peft>=0.2.0",  # For parameter-efficient fine-tuning
        "numpy>=1.20.0",
        "click>=8.0.0",  # For CLI
    ],
    entry_points={
        "console_scripts": [
            "purpose=purpose.cli:main",
        ],
    },
)
