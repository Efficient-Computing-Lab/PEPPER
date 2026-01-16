FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    apt-utils \
    git \
    curl \
    build-essential \
    cmake \
    autoconf \ 
    pkg-config \
    libtool \
    m4 \
    automake \
    gfortran \
    zlib1g-dev \
    ca-certificates \
    git \
    curl \
    wget \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install \
    blessed==1.20.0 \
    coloredlogs==15.0.1 \
    flatbuffers==25.2.10 \
    humanfriendly==10.0 \
    mpmath==1.3.0 \
    natsort==8.4.0 \
    numpy==2.2.3 \
    onnx==1.17.0 \
    onnxruntime-gpu==1.20.1 \
    opencv-python==4.11.0.86 \
    packaging==24.2 \
    protobuf==5.29.3 \
    six==1.17.0 \
    sympy==1.13.3
RUN pip3 install networkx
RUN pip3 install eventlet flask flask-socketio
RUN pip3 install lxml
COPY ./service/splitter/ /root/splitter/
COPY ./service/app.py /root/
ENTRYPOINT ["/bin/bash", "-c", "python /root/app.py"]

