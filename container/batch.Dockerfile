FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
ARG ARCH=gpu
ARG DEBIAN_FRONTEND=noninteractive
ARG py_version=3

# Validate that arguments are specified
RUN test $py_version || exit 1

RUN echo $py_version

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    nginx \
    jq \
    libsm6 \
    libxext6 \
    libxrender-dev \
    nginx \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -LO  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=3.7 numpy pyyaml scipy ipython mkl mkl-include ninja cython && \
    /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN pip install --upgrade pip

RUN python --version
RUN pip --version

# #RUN df -a

RUN pip install --trusted-host pypi.python.org -v --log /tmp/pip.log torch torchvision


# Python won’t try to write .pyc or .pyo files on the import of source modules
# Force stdin, stdout and stderr to be totally unbuffered. Good for logging
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=UTF-8 LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN nvcc --version
RUN which nvcc

RUN pip --no-cache-dir install \
    flask \
    pathlib \
    gevent \
    gunicorn \
    scipy \
    scikit-learn \
    pandas \
    fastprogress \
    python-box \
    tensorboardX


# RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext

# RUN pip --no-cache-dir install fast-bert
RUN pip install fast-bert==1.9.15

RUN pip install cryptography --upgrade && \
    pip install urllib3 --upgrade

ENV PATH="/opt/ml/code:${PATH}"
COPY /bert_batch /opt/ml/code

WORKDIR /opt/ml/code

RUN cd $WORKDIR