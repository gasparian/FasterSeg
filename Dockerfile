# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       0.4.0  (pip)
# opencv        4.0.1  (git)
# ==================================================================

FROM ubuntu:16.04
ENV LANG C.UTF-8

# ==================================================================
# tools
# ------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar

RUN git clone --depth 10 https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install

# ==================================================================
# python
# ------------------------------------------------------------------

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python


RUN DEBIAN_FRONTEND=noninteractive python -m pip --no-cache-dir install --upgrade \
        setuptools \
        requests \
        easydict==1.9 \
        protobuf==3.8.0 \
        numpy==1.16.1 \
        pandas==0.22.0 \
        Pillow==6.2.0 \
        python-dateutil==2.7.3

RUN python -m pip --no-cache-dir install --upgrade \
        scipy==1.1.0 \
        onnx==1.5.0 \
        tqdm==4.25.0 \
        # scikit-learn \
        Cython \
        # opencv-python==3.4.4.19 \
        matplotlib==3.0.0

# ==================================================================
# cityscapesScripts
# ------------------------------------------------------------------

RUN git clone --depth 10 https://github.com/mcordts/cityscapesScripts ~/cityscapesScripts && \
    cd ~/cityscapesScripts && pip install .

# ==================================================================
# opencv
# ------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
        libatlas-base-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler

RUN git clone --depth 10 --branch 3.4.5 https://github.com/opencv/opencv ~/opencv && \
    mkdir -p ~/opencv/build && cd ~/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_IPP=OFF \
          -D WITH_CUDA=OFF \
          -D WITH_OPENCL=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          .. && \
    make -j"$(nproc)" install && \
    ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2


# ==================================================================
# pytorch
# ------------------------------------------------------------------

RUN cd /tmp && \
    curl -O "https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl"  && \
    python -m pip --no-cache-dir install \
        /tmp/torch-1.1.0-cp36-cp36m-linux_x86_64.whl \
        torchvision==0.3.0


RUN python -m pip --no-cache-dir install --upgrade  \
        tensorboard==1.9.0 \
        tensorboardX==1.6 \        
        thop

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

WORKDIR /home/FasterSeg
EXPOSE 8000 6006

