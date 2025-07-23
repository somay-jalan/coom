FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install Cython packaging

RUN pip install \
    pytorch-lightning \
    omegaconf \
    hydra-core \
    pytest \
    black \
    isort \
    flake8

RUN pip install apex

RUN pip install git+https://github.com/NVIDIA/Megatron-LM

RUN git clone https://github.com/NVIDIA/NeMo.git /workspace/NeMo && \
    cd /workspace/NeMo && \
    pip uninstall -y nemo_toolkit sacrebleu && \
    pip install -e ".[nlp]" && \
    cd nemo/collections/nlp/data/language_modeling/megatron && \
    make

RUN pip install git+https://github.com/NVIDIA-NeMo/Run.git

RUN git clone https://github.com/NVIDIA/TransformerEngine.git && \
    cd TransformerEngine && \
    git submodule update --init --recursive && \
    NVTE_FRAMEWORK=pytorch \
    NVTE_WITH_USERBUFFERS=1 \
    pip install .

    
RUN pip install bitsandbytes

COPY . /workspace

CMD ["bash"]
