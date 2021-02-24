#!/bin/bash

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $dir/config.sh
export DOCKER_ARGS="-p $tb_port:6006"
$dir/run.sh tensorboard --host '0.0.0.0' --logdir /workspace/data $@
