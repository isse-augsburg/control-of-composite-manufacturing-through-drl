FROM nvcr.io/nvidia/pytorch:22.04-py3

RUN pip install stable-baselines3==1.5 \
    grequests \
    pandas \
    pillow

USER root
RUN mkdir /cfs && mkdir /sourcecode 
ENV OMP_NUM_THREADS=1

CMD /bin/bash