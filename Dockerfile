FROM nvidia/cuda:12.6.1-devel-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV COQUI_TOS_AGREED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    ninja-build \
    libgomp1 \
    libstdc++6 \
    pkg-config \
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf-2.0-dev \
    libffi-dev \
    shared-mime-info \
    ffmpeg \
    texlive \
    texlive-latex-extra \
    texlive-fonts-extra \
    texlive-latex-recommended \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --break-system-packagesnvidia-cudnn-cu12==9.1.0.70 --index-url https://pypi.org/simple

RUN pip install --no-cache-dir --break-system-packages torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple

RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir --break-system-packages llama-cpp-python==0.3.33

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt


CMD ["tail", "-f", "/dev/null"]