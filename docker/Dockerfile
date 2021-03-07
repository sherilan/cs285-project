FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

# Install system dependencies and get mujoco binaries
RUN apt-get update \
    && apt-get install -y libgl1-mesa-dev libosmesa6-dev libgl1-mesa-glx libglfw3 libglib2.0-0 \
    && apt-get install -y curl wget unzip gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
    && rm mujoco.zip \
    && touch /root/.mujoco/mjkey.txt

# Set env variable neede to compile mujoco_py
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin

# Install pip requirements
COPY requirements.txt /requirements.txt
RUN pip install --ignore-installed -r /requirements.txt

# Make it possible to import anything in workspace
ENV PYTHONPATH=$PYTHONPATH:/workspace

RUN pip install --ignore-installed pandas tensorflow tables seaborn
