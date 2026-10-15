from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nostalgic-next",
    version="1.0.0",
    author="Kaustubh Singh Rathore",
    description="A content-based movie recommendation system using TF-IDF, on-demand cosine similarity, and explainable metadata filtering.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.1.0",
        "scipy>=1.9.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "pydantic>=2.0.0",
        "requests>=2.28.0",
        "joblib>=1.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "httpx>=0.24.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
