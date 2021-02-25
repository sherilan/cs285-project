# cs285-project
Miscellaneous experiments with the DIAYN RL algorithm

## Docker Quickstart 

The [docker](docker) folder comes with a Dockerfile for building an image that should be able to run all the code in this repository. In addition, a couple of utility scripts have been added.

- [config.sh](docker/config.sh) contains all configuration variables for the docker suite 
- [build.sh](docker/build.sh) automates building of the docker image 
- [run.sh](docker/run.sh) starts a container from the built image
- [notebook.sh](docker/notebook.sh) starts a jupyter notbook server in its own container 
- [tensorboard.sh](docker/tensorboard.sh) starts a tensorboard server in its own container 

### Configuration 

Begin by editing [config.sh](docker/config.sh). By default the image name will be set to `<yourusername>/cs285`. Additionally, the path to a valid mujoco license key will default to `~/.mujoco/mjkey`, but you can change it if you keep your license somewhere else. The configuration for mapped ports (`tb_port` and `nb_port`) determines which ports (on the host machine) that will be used for tensorboard and jupyter notebooks. By default, they will map 6006:6006 (for tensorboard) and 8888:8888 (for notebooks), but if you're on a shared system, you should consider changing them to someting else to avoid conflicts. Finally, the gpu arguments will tell docker which gpus that should be exposed when running a container.

**Note**: The mujoco key must come from an institutional license since personal (hardware-bound) licenses currently don't work within docker. 

### Building 

To build the docker image, execute:

```
./docker/build.sh
```


### Running

To start a container from the image you just built, execute

```
./docker/run.sh <command -arg1 --arg2 ...>
```

If no command is provided, you will simply get an interactive bash shell. Once the command has completed (e.g. logout for the bash shell), the container will shut down and be deleted. Note that this script will also mount the root directory of this repository into the `/workspace` folder of the container. That means that any change in code on the host machine will be immediately reflected in the container, and any data written in the `/workspace` folder of the container will persist on the host after the container has been deleted.


### Tensorboard and jupyter 

Two convenience scripts have been created for starting up jupyter and tensorboard servers.

To start up a tensorboard server that will serve the content of ./data, execute:

```
./docker/tensorboard.sh
```

And to start a notebook server, execute:

```
./docker/notebook.sh
```

Once either is running, you should be able to connect to them from `localhost:<tb_port>` or `localhost:<nb_port>` (as given by [config.sh](docker/config.sh)). If you're running docker on a remote machine, traffic at the ports can be forwarded over ssh by using the `-L` option during login:

```
ssh -L <local_port1>:localhost:<tb_port> -L <local_port2>:localhost:<nb_port> username@host
```


