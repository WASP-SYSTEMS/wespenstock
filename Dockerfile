FROM ubuntu:22.04

WORKDIR /workspace

RUN apt-get -qq update && \
    apt-get -qq install \
    bear \
    build-essential \
    autoconf \
    automake \
    libtool \
    ca-certificates \
    clangd \
    curl \
    gcc \
    git \
    iputils-ping \
    jq \
    make \
    python3 \
    python3-pip \
    rsync \
    wget && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get -qq install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    git config --global --add safe.directory '*'

ARG YQ_VERSION=4.43.1
ARG YQ_BINARY=yq_linux_amd64
RUN wget -q https://github.com/mikefarah/yq/releases/download/v${YQ_VERSION}/${YQ_BINARY} -O /usr/bin/yq && \
    chmod +x /usr/bin/yq

COPY requirements.txt requirements-dev.txt ./
RUN pip3 install -r requirements.txt && \
    pip3 install -r requirements-dev.txt

ENV PYTHONUNBUFFERED=TRUE
