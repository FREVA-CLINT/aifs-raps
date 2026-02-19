#!/bin/bash --login
# (C) Copyright 2025- ECMWF.
# Modifications for DWD AI Benchmark based on the AIFS-RAPS, (C) Copyright 2026- DWD.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

#SBATCH --job-name=throughput
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH --time=4:00:00
#SBATCH -o %x-%j.out

# This script runs a training as specified in the Throughput Training Benchmark.

# The Throughput Training Benchmark evaluates the system's performance by
# running multiple 15 km-resolution model trainings simultaneously.
# Each submission of this script is a single training with `anemoi-training`.

# You may tune the number of nodes `num_nods`, the number of accelerator devices per
# node `AIFS_NUM_GPUS_PER_NODE`, the number of accelerator devices per model
# `num_gpus_per_model`, the number of CPU worker processes `num_workers`, and
# the local batch size `local_batch_size`.

PRINTENV=
if [[ -n "$PRINTENV" ]]; then
    printenv | fgrep SLURM
fi

# Submit this job from it's directory.
cd $SLURM_SUBMIT_DIR
pwd
source ../../../.aifsenv

if [[ -n "$PRINTENV" ]]; then
    printenv
fi
set -eux



num_nodes=$SLURM_NNODES # free parameter
local_batch_size=1 # free parameter
num_gpus=$(( AIFS_NUM_GPUS_PER_NODE * num_nodes )) # free parameter, you can increase this to decrease walltime via strong scaling
if [[ $local_batch_size == 1 ]]; then
    global_batch_size=16
    num_gpus_per_model=$(( num_gpus / global_batch_size ))
else
    num_gpus_per_model=1
fi

# Define the number of worker processes spawned for each accelerator device.
# More workers increases CPU memory usage but can prevent GPU stalling between iterations
num_workers=6 # Free parameter

launcher=srun # Free parameter


# Model description (fixed)
num_channels=1024  # fixed in benchmark specification
ens_size=0 # fixed in the benchmark specification
res=o800
train_end_yyyymmdd=20231230
val_start_yyyymmdd=20231231
val_end_yyyymmdd=20231231
hydra_args=
# The 10-year dataset contains 14,600 samples (10*365*4).
# Divided by batch size 16 and 2 epochs, that is 450 batches per
# model instance.
# Limiting the number of batches per model to 450 provides consistent times
# when testing with fewer parallel models, i.e. smaller batch sizes.
batch_limit=450 # FIXED
max_epochs=2 # FIXED

dataset=aifs-benchmarking-o800-10years-cloned.zarr # must be extended first:
# ./bin/aifs-clone-dataset \
#    /my-path-to/aifs-benchmarking-ea-an-oper-0001-mars-o800-2023-2023-6h-v1.zarr/ \
#    /my-path-to/aifs-benchmarking-o800-10years-cloned.zarr 14240
export CLONED_DATASET=1 # must be set when using an extended dataset

mapper_chunks=16 # FIXED in the DWD Benchmark
proc_chunks=8 # FIXED in the DWD Benchmark
# By default, training runs with the profiler
# This has an associated overhead
# You can disable it by passing the following flag
profile="--no-profile"
#profile="" #to run with the profiler
hidden_res=7
model=graphtransformer


# To verify whether your model is data-loader bound, you can
# disable data loading by setting the environment variable
# export AIFS_FAKE_DATALOADING=1
# **However, this mode should be used for diagnostic purposes only.**

jobid=$$
jobname=$SLURM_JOB_NAME

# may be used for optimization on CUDA-based systems
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Please note the benchmark constraint on the effective_global_batch_size in the documentation.
effective_global_batch_size=16 # fixed in benchmark specification
if [[ "$effective_global_batch_size" -ne "$(( num_gpus * local_batch_size / num_gpus_per_model ))" ]]; then
    echo "Parameter sanity check failed" >&2
    exit 1
fi


$RAPS_AIFS_ROOT_DIR/bin/train \
    --train-end=$train_end_yyyymmdd \
    --val-start=$val_start_yyyymmdd \
    --val-end=$val_end_yyyymmdd \
    --train-batch-limit=$batch_limit \
    --val-batch-limit=0 \
    --max-epochs=$max_epochs \
    --num-nodes="$num_nodes" \
    --num-gpus="$num_gpus" \
    --num-workers=$num_workers \
    --num-gpus-per-model="$num_gpus_per_model" \
    -C "$num_channels" \
    --model="$model" \
    --hidden-res=$hidden_res \
    -J $jobname \
    --dataset="$dataset" \
    --batch-size=$local_batch_size \
    --ens=$ens_size \
    --num-mapper-chunks=$mapper_chunks \
    --num-processor-chunks=$proc_chunks \
    --hydra-args="$hydra_args" \
    $profile \
    --launcher=$launcher \
    --res="$res"
exit $?
