# (C) Copyright 2026- DWD.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#

module load python/3.10.11

local_root_dir="$( cd $( dirname "${BASH_SOURCE[0]}" ) && cd ../../.. && pwd -P )"

#setup env
export ENV_TYPE='venv'
export INPUT_VENV_PATH="${local_root_dir:?Variable not set or empty}/venvs/aifs-raps"

export ANEMOI_BASE_SEED=42
export AIFS_DATA_PATH="/hpc/uwork/fe1ai/ecmwf_benchmark_data"

#running
export AIFS_OUTPUT_PATH=${local_root_dir}/aifs/outputs/raps
export AIFS_NUM_GPUS_PER_NODE=8
