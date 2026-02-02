"""
Setup script for fake_news_detection package.
"""

from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fake_news_detection",
    version="1.0.0",
    description="A machine learning project for detecting fake news using TF-IDF and Passive Aggressive Classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Timofey Krylov",
    author_email="timofey.krylov.0206@gmail.com",
    url="https://github.com/GZGef/fake_news_detection",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=requirements,
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="fake-news-detection machine-learning nlp tf-idf passive-aggressive-classifier",
    project_urls={
        "Bug Reports": "https://github.com/GZGef/fake_news_detection/issues",
        "Source": "https://github.com/GZGef/fake_news_detection",
    },
)