#!/bin/bash

# Get current dir and source config
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $dir/config.sh --echo


# Check valid mujoco path
if [ ! -f "$mj_key" ]; then
  echo "ERROR: Mujoco license not found at $mj_key"
  echo "Specify an alternative source with the MJKEY environment variable"
  exit
fi

# Check if specific arguments are given
if [ -z "$1" ]; then
  cmd="bash"
  echo "Executing bash shell"
else
  cmd=$@
  echo "Executing custom command: $cmd"
fi

# Get extra docker args specified by caller
docker_args=$DOCKER_ARGS

# Generate docker command
docker run --rm -it --gpus $gpus \
  -v $mj_key:/root/.mujoco/mjkey.txt \
  -v $dir/..:/workspace \
  $docker_args $image $cmd
