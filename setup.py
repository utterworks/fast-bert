import sys
from io import open
from setuptools import setup, find_packages
import subprocess


with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

# get version from VERSION.txt 
with open("VERSION") as f:
    version = f.read().strip()

setup(
    name="fast_bert",
    # get version from VERSION file
    version=version,
    description="AI Library using BERT",
    author="Kaushal Trivedi",
    author_email="kaushaltrivedi@me.com",
    license="Apache2",
    url="https://github.com/kaushaltrivedi/fast-bert",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="BERT NLP deep learning google",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
