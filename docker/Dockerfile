
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

# Copy mujoco binaries and requirements
COPY mujoco /root/.mujoco
COPY requirements.txt /requirements.txt

# Set env variable neede to compile mujoco_py
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin

# Install dependencies
RUN apt-get update && \
    apt-get install -y gcc && \
    apt-get install -y libgl1-mesa-dev libosmesa6-dev libgl1-mesa-glx libglfw3 libglib2.0-0 && \
    pip install --ignore-installed -r /requirements.txt

# Make it possible to import anything in workspace
ENV PYTHONPATH=$PYTHONPATH:/workspace
