# FROM pytorch/pytorch:1.8.0-cuda11.0-cudnn8-devel
FROM pytorchlightning/pytorch_lightning:base-cuda-py3.8-torch1.7

COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install -r requirements.txt
WORKDIR /