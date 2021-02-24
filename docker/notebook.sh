#!/bin/bash

# Get current dir and source config
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $dir/config.sh --echo
docker exec -it $container jupyter notebook --allow-root --ip='0.0.0.0' $@
