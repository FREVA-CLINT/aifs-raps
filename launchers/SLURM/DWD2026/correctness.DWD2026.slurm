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

#SBATCH --job-name=correctness
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=1:00:00
#SBATCH -o %x-%j.out

# This script runs the Correctness Test Benchmark.
# This job must be run using 4 devices to produce the expected result.
# A different number of devices changes the order in which data samples
# are loaded and might affect the final loss.

# The job training should take about 10 to 30 minutes.

# After completion, a small utility script `bin/test-correctness <rundir>`
# verifies that the training loss has converged to loss < 0.007 . The utility
# script takes a list of rundirs and returns a pass or fail result.
# The rundir is printed in the job std-out.

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

num_nodes=$SLURM_NNODES
num_gpus=4 #FIXED
num_gpus_per_model=1 #FIXED
effective_global_batch_size=16 #FIXED
local_batch_size=$((effective_global_batch_size / num_gpus))
launcher=""

# Model description (fixed)
num_channels=256
ens_size=0
res=o48
export ANEMOI_BASE_SEED=42 # FIXED
train_end_yyyymmdd=2020
val_start_yyyymmdd=2021
val_end_yyyymmdd=2021
hydra_args="training.max_steps=30000"
hydra_args+=" training.lr.rate=1e-3"
hydra_args+=" training.lr.iterations=30000"
hydra_args+=" training.lr.min=3e-7"
hydra_args+=" model.processor.window_size=256"
batch_limit=1000
val_batch_limit=100
max_epochs=5
dataset=aifs-ea-an-oper-0001-mars-o48-1979-2022-6h-v6.zarr # Available for download
mapper_chunks=4
proc_chunks=8
num_workers=8
hidden_res=o32
model=transformer

jobid=$$
jobname=$SLURM_JOB_NAME

# may be used for optimization on CUDA-based systems
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$RAPS_AIFS_ROOT_DIR/bin/train \
    -j "$jobid" \
    --train-end=$train_end_yyyymmdd \
    --val-start=$val_start_yyyymmdd \
    --val-end=$val_end_yyyymmdd \
    --train-batch-limit=$batch_limit \
    --val-batch-limit=$val_batch_limit \
    --max-epochs=$max_epochs \
    --num-nodes="$num_nodes" \
    --num-gpus="$num_gpus" \
    --num-workers=$num_workers \
    --num-gpus-per-model="$num_gpus_per_model" \
    -C "$num_channels" \
    --model="$model" \
    --hidden-res=$hidden_res \
    -J "$jobname" \
    --dataset="$dataset" \
    --batch-size $local_batch_size \
    --num-mapper-chunks=$mapper_chunks \
    --num-processor-chunks=$proc_chunks \
    --hydra-args="$hydra_args" \
    --launcher=$launcher \
    --res="$res" \
    --no-profile
exit $?
