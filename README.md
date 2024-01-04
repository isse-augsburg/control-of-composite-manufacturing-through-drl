# Control of Composite Manufacturing through Deep Reinforcement Learning

Repository for both RL code and RTM Simulation for the Paper "Control of Composite Manufacturing through Deep Reinforcement Learning", accepted at the International Conference on Machine Learning and Applications (ICMLA) 2023.

#### Requirements/ Installation:

Both module simply require docker. If you want to create own datasets and do further experiments, local julia and python installations and further programming knowledge are helpful.


#### How set up the RtmServer:


- config file:

    The RtmServer needs a configuration file of the following format. An example file is provided at `simulation_storage/configs/config.csv`

    csv-file of format:

            t_step,p_min,p_max,fvc_normal,fvc_patch,filelist,data_source,inlets,base_mesh
            0.5,10000,500000,0.35,0.45,<filelist>,<data_source>,3,<base_mesh>

    ##### filelist:

    Name of the filelist to select from during validation. Has to be the name of a .jld2 file.
    The file extension will be appended automatically.
    When in eval mode the server will reset to files [1, nenvs] (deterministic) when calling reset(). 
    Each single environment is basically useless after finishing once, because it will reset to a randomly selected file out of the given filelist.
    This behaviour is meant only for validation during training.
    If you want to create own datasets through some jula scripting:
    `filelist.jld2` needs to be a `Vector{String}`, written to .jld2.
    The entries are the raw paths to the `<mesh-xxx>.jld2` files of the dataset.
    Meshes can be created by manually calling functions from `MeshGnerator.jl`, where you can also set parameters for patch placement, size, etc.

    ###### data_source:

    Path that will be prepended to filenames when invoking GET /file_select
    Usually used to evaluate models on a testset.
    Needs to end with / or \\, dependent on the target os.

    ###### inlets: 

    Number of inlets distributed on the left side of the part (see paper). Config.jl contains sets of mesh elements that support the numbers 3, 5, 7.
    Other inlet configurations would require you to determine the mesh element set corresponding to each inlet manually.

    ###### base_mesh:

    Path to the mesh of the part in .ERFH5 format (used for random case generation in training). Theoretically all sorts of 2D meshes can be used, practically the mapping of element ids to inlets and the plotting would have to be adjusted.

- move to the RtmServer directory and build the docker image via:

        docker build -t rtm-server .

- start the container:

    The container can be started via docker run. 
    The name of the config file located within `/sim_storage/configs/` needs to be provided with:

        -e CONFIG_FILE=

    The sourcecode directory needs to mounted with:

        -v <YOUR REPO PATH>/control-of-composite-manufacturing-through-drl/RtmServer/src:/sourcecode

    The shared directory needs to mounted with:

        -v <YOUR REPO PATH>/control-of-composite-manufacturing-through-drl/simulation_storage:/cfs

    Map a port from your host system to the docker container:

        -p <PORT>:8080


    For example:

        docker run  --name rtm-server-1 -p 8080:8080 -e NENVS=2 -e CONFIG_NAME="config.csv" -v /Users/XXX/repos/control-of-composite-manufacturing-through-drl/simulation_storage:/cfs -v /Users/XXX/repos/control-of-composite-manufacturing-through-drl/RtmServer/src:/sourcecode rtm-server

#### How to set up the RlClient module
Move to the RlClient directory and build the docker image via (takes quite long):

        docker build -t rl-client .

If you encounter OPEN_BLAS Warnings when executing code inside the container, enter `export OMP_NUM_THREADS=1` to the shell.

#### Example data and config:

Example data, a trained model and a config file are provided at `./similation_storage/...` and are to be mounted into the docker containers at `/cfs`.
The provided test and validation set refer to the experiment presented in the paper and the config is set up accordingly.
The model checkpoints are taken from the A2C/PPO agents trained with flowfront, pressure and fvc observation.

#### Training:

- Start at least two servers (one is needed for mid training validation):

        docker run  --name rtm-server-1 -p 8080:8080 -e NENVS=2 -e CONFIG_NAME="config.csv" -v <YOUR PATH>/control-of-composite-manufacturing-through-drl/simulation_storage:/cfs -v <YOUR PATH>/control-of-composite-manufacturing-through-drl/RtmServer/src:/sourcecode rtm-server

        docker run  --name rtm-server-2 -p 8081:8080 -e NENVS=2 -e CONFIG_NAME="config.csv" -v <YOUR PATH>/control-of-composite-manufacturing-through-drl/simulation_storage:/cfs -v <YOUR PATH>/control-of-composite-manufacturing-through-drl/RtmServer/src:/sourcecode rtm-server

- Run the clientside docker container:

        docker run --name rl-client -it -v <YOUR PATH>/control-of-composite-manufacturing-through-drl/simulation_storage:/cfs -v /Users/leoheber/repos/control-of-composite-manufacturing-through-drl/RlClient/src:/sourcecode rl-client bash

    Attach a shell or a Visual Studio Code window to the running container.
    Use the script at `/sourcecode/example_evaluation.py` (your local `.../RlClient/src/` mounted in the container)
    Execute the script under RlClient/src/example_training.py
    Insert the IPs the RtmServers are serving on.
    Adjust the envs_per_server variable.
    Insert your desired save paths in the python script (use something from your mounted `/cfs/results/`` directory).
    You can adjust some parameters in the training script, those are marked with `# TODO`.
    Note that a high number of steps is needed for meaningful training results, which can be hard to achieve on a single machine.

#### Evaluation:

- Start a server:

        docker run  --name rtm-server-1 -p 8080:8080 -e NENVS=8 -e CONFIG_NAME="config.csv" -v <YOUR PATH>/control-of-composite-manufacturing-through-drl/simulation_storage:/cfs -v /Users/leoheber/repos/control-of-composite-manufacturing-through-drl/RtmServer/src:/sourcecode rtm-server

- Run the clientside docker container:

        docker run --name rl-client -it -v <YOUR PATH>/control-of-composite-manufacturing-through-drl/simulation_storage:/cfs -v /Users/leoheber/repos/control-of-composite-manufacturing-through-drl/RlClient/src:/sourcecode rl-client bash

    Attach a shell or a Visual Studio Code window to the running container.
    Use the script at `/sourcecode/example_evaluation.py` (your local `.../RlClient/src/`` mounted in the container).
    Insert the IP the RtmServer is serving on.
    Adjust the `envs_per_server` variable.
    Insert your desired save paths and the path to the model to evaluate.
    Execute the script.
