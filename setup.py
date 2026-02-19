#!/usr/bin/env python3
from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "readme.md").read_text(encoding="utf-8")

setup(
    name="llm-fingerprinter",
    version="0.1.0",
    author_email="maxmassi12@gmail.com",
    url="https://github.com/litemars/llm-fingerprinter",
    description="Black-box LLM fingerprinting system for model identification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="litemars",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "llm-fingerprinter=cli:cli",
        ],
    },
    install_requires=[
        line.strip()
        for line in (this_directory / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ],
)
