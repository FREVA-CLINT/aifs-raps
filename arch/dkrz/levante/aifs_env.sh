# (C) Copyright 2026- DWD.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#

# Adjust this module line to whatever Python module is available on Levante.
module load python3

local_root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../../.. && pwd -P )"

# Setup env
export ENV_TYPE='venv'
export INPUT_VENV_PATH="${local_root_dir}/venv"

export ANEMOI_BASE_SEED=42

# Optional: point this at a shared dataset root if you have one on Levante.
# export AIFS_DATA_PATH="/path/to/aifs-data"

# Runtime settings
export AIFS_OUTPUT_PATH="${local_root_dir}/aifs/outputs/raps"

# Update this to the actual GPU count per node on Levante for your partition.
export AIFS_NUM_GPUS_PER_NODE=4
