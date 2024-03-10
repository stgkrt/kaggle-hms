ARG CONTAIER_VER=140

FROM gcr.io/kaggle-gpu-images/python:v${CONTAIER_VER}

ENV lang="ja_jp.utf-8" language="ja_jp:ja" lc_all="ja_jp.utf-8"

# when you cannot use ssh
ENV LD_LIBRARY_PATH=/usr/local/lib

RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install hydra-core segmentation-models-pytorch

WORKDIR /kaggle
