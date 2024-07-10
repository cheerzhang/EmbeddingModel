#!/usr/bin/env python
# coding=utf-8
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='emb-model',
    version='0.0.1',
    author="ZhangLe",
    author_email="zhangle@gmail.com",
    description="simple useing for embedding models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cheerzhang/EmbeddingModel",
    project_urls={
        "Bug Tracker": "https://github.com/cheerzhang/EmbeddingModel/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages("."),
    install_requires=[
        'torch>=2.3.1'
    ],
    python_requires=">=3.11.2",
)