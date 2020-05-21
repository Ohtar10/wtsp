FROM conda/miniconda3

LABEL maintainer="Luis Eduardo Ferro Diez <luisedof10@gmail.com>"
LABEL product="Where To Sell Products (wtsp)"

ENV LC_ALL=C.utf-8
ENV LANG=C.utf-8

# Add assets
COPY wtsp-0.1.1.tar.gz wtsp-0.1.1.tar.gz 

# Update the base conda environment according to yaml file
RUN conda update -y -n base -c defaults conda
RUN conda create -n wtsp python=3.7 pip
RUN conda init bash && \
    echo "export LC_ALL=C.UTF-8" >> ~/.bashrc && \
    echo "export LANG=C.UTF-8" >> ~/.bashrc && \
    echo "source activate wtsp" >> ~/.bashrc
ENV PATH /usr/local/envs/wtsp/bin:$PATH

# Install wtsp tool
RUN pip install /wtsp-0.1.1.tar.gz