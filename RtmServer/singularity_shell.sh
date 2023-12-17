

export SINGULARITY_DOCKER_USERNAME=\$oauthtoken
export SINGULARITY_DOCKER_PASSWORD=ZzhiZHVoaGIzNGNtMmFmY2dmZ3YzMDEwdmw6OWQwYTY0MjgtMjc2OS00ZTY5LWI3ZjYtOWNmM2RlMjIwM2Ew

export PYTHONPATH="${PYTHONPATH}:/cfs/home/h/e/heberleo/BA"


srun --pty --partition=big-cpu --mem=10000 --ntasks=10 singularity shell --contain -B /cfs:/cfs docker://nvcr.io/isse/pytorch_julia:22.04-py3