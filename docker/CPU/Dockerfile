FROM pytorchlightning/pytorch_lightning:base-conda-py3.8-torch1.7

COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install -r requirements.txt
WORKDIR /
RUN conda deactivate