#!/bin/bash

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $dir/config.sh
export DOCKER_ARGS="-p $nb_port:8888"
$dir/run.sh jupyter notebook --allow-root --ip='0.0.0.0' $@
