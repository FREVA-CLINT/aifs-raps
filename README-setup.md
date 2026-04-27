# Content

This repository contains DKRZ launcher scripts and helper tools to build AIFS-RAPS and run GPU benchmark trainings.

The benchmark campaign in this README contains 4 runs:
- correctness benchmark on 1 node
- throughput benchmark on 2 nodes with real data loading (`AIFS_FAKE_DATALOADING=0`)
- throughput benchmark on 2 nodes with fake data loading (`AIFS_FAKE_DATALOADING=1`)
- throughput benchmark on 4 nodes with fake data loading (`AIFS_FAKE_DATALOADING=1`)

All relevant SLURM files are in `launchers/SLURM/DKRZ2026/`.

# Build

1. Clone and enter the repository:

```bash
cd aifs-raps
```

2. Create a configuration for your system: `arch/foo/bar/aifs_env.sh`. Use `arch/dkrz/hlre5/aifs_env.sh` for python3.13.

2.1. Optional: set an explicit Python executable in the selected `aifs_env.sh`:

```bash
export INPUT_PYTHON_EXECUTABLE="/path/to/python3"
```
2.2. Set the root path of your datasets:

```bash
export INPUT_PYTHON_EXECUTABLE="/path/to/python3"
```

4. Build AIFS-RAPS:

```bash
./aifs-build --build-dir build --arch arch/foo/bar
```

5. Load the generated runtime environment:

```bash
source ./.aifsenv
```

# Inputs

Two datasets are required.

1. Correctness dataset:
- `aifs-ea-an-oper-0001-mars-o48-1979-2022-6h-v6.zarr`

2. Throughput dataset:
- `aifs-benchmarking-o800-10years-cloned.zarr`
- This must be extended first with:

```bash
./bin/aifs-clone-dataset \
  /my-path-to/aifs-benchmarking-ea-an-oper-0001-mars-o800-2023-2023-6h-v1.zarr/ \
  /my-path-to/aifs-benchmarking-o800-10years-cloned.zarr \
  14240
```

Place both datasets under your data root and point `AIFS_DATA_PATH` in the aifs_env.sh to that root.

# Run

Run all commands from:

```bash
cd launchers/SLURM/DKRZ2026
```

## 1) Correctness benchmark (single node)

Use:
- `correctness.DKRZ2026.slurm`

Submit:

```bash
sbatch correctness.DKRZ2026.slurm
```

After the run, extract the run directory and validate correctness:

```bash
rg "rundir =" logs/correctness-<jobid>.out
../../bin/test-correctness <rundir>
```

## 2) Throughput benchmark, 2 nodes, real data loading

Use:
- `throughput.DKRZ2026.real.2nodes.slurm`

Submit:

```bash
sbatch throughput.DKRZ2026.real.2nodes.slurm
```

## 3) Throughput benchmark, 2 nodes, fake data loading

Use:
- `throughput.DKRZ2026.fake.2nodes.slurm`

Submit:

```bash
sbatch throughput.DKRZ2026.fake.2nodes.slurm
```

## 4) Throughput benchmark, 4 nodes, fake data loading

Use:
- `throughput.DKRZ2026.fake.4nodes.slurm`

Submit:

```bash
sbatch throughput.DKRZ2026.fake.4nodes.slurm
```

# Validation

Validate the correctness run with [test-correctness](bin/test-correctness).  
The validation is successful only if the script output contains `PASS`.

```bash
../../../bin/test-correctness <rundir>
```

Validate the throughput runs (real 2-node, fake 2-node, fake 4-node) with [aifs-jobinfo](bin/aifs-jobinfo):

```bash
../../../bin/aifs-jobinfo <rundir-real-2nodes> <rundir-fake-2nodes> <rundir-fake-4nodes>
```
