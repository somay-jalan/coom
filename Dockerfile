FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    Cython==3.0.8 \
    packaging==23.2

RUN pip install \
    pytorch-lightning==2.5.2 \
    omegaconf==2.3.0 \
    hydra-core==1.3.2 \
    pytest==8.1.1 \
    black==24.3.0 \
    isort==5.13.2 \
    flake8==7.0.0

RUN pip install apex==0.1

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
    git checkout release_v2.6 && \
    git submodule update --init --recursive && \
    NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 pip install .

RUN pip install bitsandbytes==0.46.0

COPY . /workspace

CMD ["bash"]
