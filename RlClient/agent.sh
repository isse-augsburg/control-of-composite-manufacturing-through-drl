#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=RL4RTM_client
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10000
#SBATCH --output=/cfs/home/h/e/heberleo/BA/server_logs/client-%A.out

export SINGULARITY_DOCKER_USERNAME=\$oauthtoken
export SINGULARITY_DOCKER_PASSWORD=ZzhiZHVoaGIzNGNtMmFmY2dmZ3YzMDEwdmw6OWQwYTY0MjgtMjc2OS00ZTY5LWI3ZjYtOWNmM2RlMjIwM2Ew

export PYTHONPATH="${PYTHONPATH}:/cfs/home/h/e/heberleo/BA"

singularity exec --nv -B /cfs:/cfs docker://nvcr.io/isse/pytorch_julia:22.04-py3 python3 /cfs/home/h/e/heberleo/BA/RlClient/src/training.py