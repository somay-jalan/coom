FROM nvcr.io/nvidia/pytorch:23.12-py3

RUN pip install --upgrade pip setuptools wheel && \
    pip install \
      pytorch-lightning==1.8.6 \
      omegaconf \
      hydra-core \
      pytest \
      black \
      isort \
      flake8

RUN git clone https://github.com/NVIDIA/apex /tmp/apex && \
    cd /tmp/apex && \
    pip install --no-cache-dir . --global-option="--cpp_ext" --global-option="--cuda_ext" && \
    cd / && \
    rm -rf /tmp/apex

RUN pip install git+https://github.com/NVIDIA/Megatron-LM.git@114fabee9fdb0e4b1f3d1cfae73e824a3d16fdc4

RUN pip install 'nemo_toolkit[all]'

CMD ["bash"]
