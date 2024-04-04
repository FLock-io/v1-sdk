FROM nvidia/cuda:12.2.0-base-ubuntu22.04
#FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# ================ Install Python from source code =================

RUN apt-get update && apt-get install -y \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    wget \
    libbz2-dev \
    liblzma-dev \
    git

# Install and compile specific version (Python 3.11.3)
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.11.3/Python-3.11.3.tgz
RUN tar -xvf Python-3.11.3.tgz
WORKDIR /tmp/Python-3.11.3
RUN ./configure --enable-optimizations
RUN make altinstall

# cleanup
WORKDIR /
RUN rm -r /tmp/Python-3.11.3

# ================ Install Python from source code ================

RUN python3.11 -m pip install --upgrade pip

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000/tcp
CMD [ "python3.11", "./main.py", "--conf", "conf.yaml" ]

