# Control of Composite Manufacturing through Deep Reinforcement Learning

Repository for both RL code and RTM Simulation for the Paper "Control of Composite Manufacturing through Deep Reinforcement Learning", accepted at the International Conference on Machine Learning and Applications (ICMLA) 2023.

Requirements/ Installation:

The RtmServer module simply requires docker. If you want to create own datasets and do further experiments, you can start the provided image as an interactive session, so you don't have to install Julia.

The RlClient module requires python with some packages installed. See the provided TODO

How to start the RtmServer:

- shared directory:
    The Python/ RL - software and the Julia/ simulation - software exchange data through a common directory. The data within is not meant for user interaction. You only need to set the path (will be explained later) in both programs and make sure that both can access it. We used a server cluster with a shared filesystem in our experiments. Obviously, executing everything on one machine also works (i.e. for evaluation).

- move to the RtmServer directory ad build the docker image via:

    docker build -t rtm-server .


- config file:

    The RtmServer needs a configuration file of the following format. An example file is provided at RtmServer/configs.config.csv

    csv-file of format:

        t_step,p_min,p_max,fvc_normal,fvc_patch,filelist,data_source,inlets,base_mesh
        0.5,10000,500000,0.35,0.45,<filelist>,<data_source>,3,<base_mesh>

        filelist:

        Name of the filelist to select from during validation. Has to be the name of a .jld2 file.
        The file extension will be appended automatically.
        Filelists can be created with CreateMesh.jl
        When in eval mode the server will reset to files [1, nenvs] (deterministic) when calling reset(). 
        Each single environment is basically useless after finishing once, because it will reset to a randomly selected file out of the given filelist.
        This behaviour is meant only for validation during training.

        data_source:

        Path that will be prepended to filenames when invoking GET /file_select
        Usually used to evaluate models on a testset.
        Needs to end with / or \, dependent on the target os.

        inlets: 

        Number of inlets distributed on the left side of the part (see paper). Config.jl contains sets of mesh elements that support the numbers 3, 5, 7.
        Other inlet configurations would require you to determine the mesh element set corresponding to each inlet manually.

        base_mesh:

        Path to the mesh of the part in .ERFH5 format (used for random case generation in training). Theoretically all sorts of 2D meshes can be used, practically the mapping of element ids to inlets and the plotting would have to be adjusted.

- start the container:

    docker run  --name rtm-server-container -p <Port you want to use>:8080 -e NENVS=<Number of simulation workers> -e CONFIG_PATH=<Path to your config file> -v <Path to your shared directory>:/cfs/file_exchange rtm-server


Training:

Execute the script under RlClient/src/example_training.py

Evaluation:

Execute the script under RlClient/src/example_evaluation.py

