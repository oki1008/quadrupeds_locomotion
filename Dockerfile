# RTX 5080用のベース
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# 基本ツールのインストール
RUN apt-get update && apt-get install -y \
    git build-essential python3.10 python3.10-dev python3-pip wget \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libegl1-mesa \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# RTX 5080対応のPyTorchをインストール
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspace

# お掃除済みリストを使って小物ライブラリをインストール
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# rsl_rlとGenesisのインストール　（Version 1.0.2）
RUN git clone https://github.com/leggedrobotics/rsl_rl.git /tmp/rsl_rl && \
    cd /tmp/rsl_rl && \
    git checkout v1.0.2 && \
    pip install -e .
RUN pip install --no-cache-dir "genesis-world[gui]" torchlibrosa

CMD ["/bin/bash"]