#!/usr/bin/env python
# coding=utf-8
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='emb-model',
    version='0.1.12',
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
    include_package_data=True,
    package_data={
        'emb_model': ['resources/*.yaml'],
    },
    install_requires=[
        'torch==2.3.1',
        'scikit-learn==1.5.1',
        'numpy==1.26.4',
        'pandas==2.1.4',
        'mlflow==2.11.3'
    ],
    python_requires=">=3.9.0",
)