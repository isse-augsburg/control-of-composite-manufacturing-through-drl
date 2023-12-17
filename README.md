# Control of Composite Manufacturing through Deep Reinforcement Learning

Repository for both RL code and RTM Simulation for the Paper "Control of Composite Manufacturing through Deep Reinforcement Learning", accepted at the International Conference on Machine Learning and Applications (ICMLA) 2023.

Installation:

Code setup:

The code does not run completeley out of the box, a few things need to be done to set up the software. Mainly some paths need to be set.

- shared directory:
    The Python/ RL - software and the Julia/ simulation - software exchange data through a common directory. The data within is not meant for user interaction. You only need to set the path (will be explained later) in both programs and make sure that both can access it. We used a server cluster with a shared filesystem in our experiments. Obviously, executing everything on one machine also works (i.e. for evaluation).

- simulation server setup:
    The file RtmServer/src/Paths.jl needs to be adjusted to the right paths.

    - set <shared_fs> to the aforementioned shared directory path.
    - set <sim_storage> to a path to a directory, where internal simulation data can be stored.

    Don't remove the shared keyword before the variable name!

- simulation config:
        

Data requirements and preparation:

Training:

Evaluation:

Utilities:
