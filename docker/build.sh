#!/bin/bash

# Get current dir
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $dir/config.sh --echo

# Change into dir and build
cd $dir
docker build --force-rm -t $image .
