FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive OPENCV_IO_ENABLE_OPENEXR=1 GRADIO_ANALYTICS_ENABLED=False
RUN env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy apt-get -o Acquire::http::Proxy=DIRECT -o Acquire::https::Proxy=DIRECT update && env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy apt-get -o Acquire::http::Proxy=DIRECT -o Acquire::https::Proxy=DIRECT install -y --no-install-recommends git python3.10 python3.10-venv python3-pip libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m venv /opt/f2g-env
ENV PATH=/opt/f2g-env/bin:$PATH
WORKDIR /app
COPY . /app
ARG TORCH_CUDA_ARCH_LIST="8.9;9.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=4
RUN env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    PIP_CONFIG_FILE=/dev/null NO_PROXY='*' no_proxy='*' sh -c \
    'pip --proxy="" install --no-cache-dir -r requirements-lock-cu124.txt && \
     pip --proxy="" install --no-build-isolation --no-deps ./third_party/pointnet2_ops ./third_party/chamfer && \
     pip --proxy="" install --no-deps --no-build-isolation .'
RUN env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy git -c http.proxy= -c https.proxy= clone https://github.com/NVlabs/RADIO.git /opt/RADIO && git -C /opt/RADIO checkout fbd19ec1e68483482d7d96a59cb639880a8b33ed
ENV F2G_RADIO_REPOSITORY=/opt/RADIO F2G_RADIO_CHECKPOINT=/weights/radio-v2.5-l_half.pth.tar
EXPOSE 7860
CMD ["f2g-pose", "demo", "--checkpoint", "/weights/f2g-pose.pt", "--host", "0.0.0.0", "--port", "7860", "--output-root", "/outputs"]
