# (C) Copyright 2026- DWD.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#

# Load Python module unless caller provides an explicit executable.
# Example:
#   export INPUT_PYTHON_EXECUTABLE="/sw/.../python3"
module load python
module load git

local_root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../../.. && pwd -P )"

# Setup env
export ENV_TYPE='venv'
export INPUT_VENV_PATH="${local_root_dir}/venv-hlre5"
export ANEMOI_BASE_SEED=42

# set the correct data path
export AIFS_DATA_PATH=/work/bd1560/k204233/benchmarking/data/

# set the correct anemoi-datasets version, with python3.12 it will install incompatible 0.5.36
export INPUT_ANEMOI_DATASETS_VERSION="0.5.35"
export PIP_IGNORE_REQUIRES_PYTHON=1

# Runtime settings
export AIFS_OUTPUT_PATH="${local_root_dir}/aifs/outputs/raps"

# Update this to the actual GPU count per node for your partition.
export AIFS_NUM_GPUS_PER_NODE=4
