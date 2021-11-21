FROM jupyter/scipy-notebook:33add21fab64

RUN pip3 install torch==1.10 torchvision torchaudio
