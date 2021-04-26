FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN pip install scipy==1.3.3
RUN pip install requests==2.22.0
RUN pip install Pillow==6.2.1
RUN pip install tqdm
RUN pip install cmake
RUN pip install dlib

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    imagemagick libmagickwand-dev --no-install-recommends \
 && rm -rf /var/lib/apt/lists/*

# Setup CLIP
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install ftfy regex tqdm
RUN pip install git+https://github.com/openai/CLIP.git

WORKDIR /content
COPY . .