from io import open
from setuptools import setup, find_packages

setup(name='fast_bert',
      version='0.1',
      description='AI Library using BERT',
      author='Kaushal Trivedi',
      long_description=open("README.md", "r", encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      keywords='BERT NLP deep learning google',
      packages=find_packages(exclude=["*.tests", "*.tests.*",
                                      "tests.*", "tests"]),
      install_requires=[
          'pytorch-pretrained-bert',
          'fastai'
      ],
      zip_safe=False)
