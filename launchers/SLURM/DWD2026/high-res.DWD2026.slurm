#!/bin/bash --login
# (C) Copyright 2025- ECMWF.
# (C) Copyright 2026- DWD.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

#SBATCH --job-name=high-res
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH --time=4:00:00
#SBATCH -o %x-%j.out

# This script runs a high-resolution training as specified in the High-resolution Training Benchmark

# This benchmark test aims to demonstrate the system's capability to
# execute a high-resolution, large-scale training job.

# You may tune the number of nodes `num_nods`, the number of accelerator devices per
# node `AIFS_NUM_GPUS_PER_NODE`, the number of accelerator devices per model
# `num_gpus_per_model`, and the number of CPU worker processes `num_workers`

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
num_gpus=$((AIFS_NUM_GPUS_PER_NODE * num_nodes)) # free parameter
num_gpus_per_model=4 # free parameter, controls model sharding.
# Please note the benchmark constraint on the effective_global_batch_size_per_ensmember in the documentation.


# Define the number of worker processes spawned for each accelerator device.
# More workers increases CPU memory usage but can prevent GPU stalling between iterations
num_workers=6 # Free parameter

launcher=srun # Free parameter

# Model description (fixed)
num_channels=2048 # fixed in the benchmark specification
ens_size=4 # FIDEX
res=o2560
train_end_yyyymmdd="20230929"
val_start_yyyymmdd="20230930"
val_end_yyyymmdd="20230930"
hydra_args="data=zarr_no_insolation"
# The 2-year dataset contains 2920 samples (2*365*4).
# Divided by batch size 16 and 2 epochs, that is 90 batches per
# model instance and ensemble member.
# Limiting the number of batches per model to 90 provides consistent times when
# testing with fewer parallel models, i.e. smaller batch sizes and ensemble.
batch_limit=90 # FIXED
max_epochs=2 # FIXED

dataset=aifs-benchmark-4km-2years-synthetic.zarr # must be extended first:
# ./bin/aifs-clone-dataset \
#    /my-path-to/aifs-rd-an-lwda-ifc3-mars-o2560-2023-2023-6h-v3-1week.zarr \
#    /my-path-to/aifs-benchmark-4km-2years-synthetic.zarr 2920
export CLONED_DATASET=1 # must be set when using an extended dataset

mapper_chunks=32 # FIXED in the DWD Benchmark
proc_chunks=8 # FIXED in the DWD Benchmark
# By default, training runs with the profiler
# This has an associated overhead
# You can disable it by passing the following flag
profile="--no-profile"
#profile="" #to run with the profiler
hidden_res=o256
model=transformer
local_batch_size=1 # fixed

# To verify whether your model is data-loader bound, you can
# disable data loading by setting the environment variable
# export AIFS_FAKE_DATALOADING=1
# **However, this mode should be used for diagnostic purposes only.**

jobid=$$
jobname=$SLURM_JOB_NAME

# may be used for optimization on CUDA-based systems
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Please note the benchmark constraint on the effective_global_batch_size_per_ensmember in the documentation.
num_gpus_per_ensmember=$(( num_nodes * AIFS_NUM_GPUS_PER_NODE / ens_size ))
effective_global_batch_size_per_ensmember=16 # fixed in benchmark specification
if [[ "$effective_global_batch_size_per_ensmember" -ne "$(( num_gpus_per_ensmember * local_batch_size / num_gpus_per_model ))" ]]; then
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
