ARG MAYBE_GPU
FROM tensorflow/tensorflow:1.15.2${MAYBE_GPU}-py3-jupyter

RUN pip install --no-cache-dir --upgrade pip==19.3.1

# fasttext
ADD https://github.com/facebookresearch/fastText/archive/v0.9.2.tar.gz /fasttext.tar.gz
RUN cd / && tar zxf fasttext.tar.gz && cd fastText-0.9.2 && pip install --no-cache-dir .

COPY requirements.txt /requirements.txt 
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY data /data
COPY src /src

ENV PYTHONPATH /src
WORKDIR /src

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/src --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token=''"]
