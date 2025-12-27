setup.py"""Setup script for FinanceInsight package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="FinanceInsight",
    version="0.1.0",
    author="shivraj1182",
    author_email="your.email@example.com",
    description="NER models for financial data extraction from unstructured documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shivraj1182/FinanceInsight",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "financeinsight=src.cli:main",
        ],
    },
    include_package_data=True,
    keywords="NER financial-nlp named-entity-recognition transformer BERT",
)
