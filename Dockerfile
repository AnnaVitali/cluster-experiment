FROM nvcr.io/nvidia/modulus/modulus:24.04

ARG USER_NAME
WORKDIR /home/${USER_NAME}/persistent

RUN git clone https://github.com/AnnaVitali/cluster-experiment.git

WORKDIR /home/${USER_NAME}/persistent/cluster-experiment/forward_top_bottom_lamp_simulation

CMD ["python", "train.py"]