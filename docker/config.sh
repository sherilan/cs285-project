
# Configs for running project with docker

# Name of image: default = <username>/cs285
image=${IMAGE:-"${USER}/cs285"}
# Name of container: default = <username>_cs285
container=${CONTAINER:-"${USER}_cs285"}
# Path to mujoco key: default = ~/.mujoco/mjkey.txt
mj_key=${MJ_KEY:-$HOME/.mujoco/mjkey.txt}
# Host port for tensorboard
tb_port=${TB_PORT:-6006}
# Host port for jupyter
nb_port=${NB_PORT:-8888}
# Gpus to run with
gpus=${GPUS:-all}

if [ "$1" == "--echo" ]; then
  echo "Using image name: $image"
  echo "Using container name: $container"
  echo "Using mujoco licence key: $mj_key"
  echo "Using port mapping (tb) $tb_port:6006"
  echo "Using port mapping (nb) $nb_port:8888"
  echo "Using gpus: $gpus"
fi
