#!/usr/bin/env bash
# RL rollout over NVSHMEM-CXL, 2 PEs on the 5090 with the symmetric heap
# bound to the CXL NUMA node.
#
#   ./run.sh                     cxl mode, defaults
#   ./run.sh --mode vram         conventional copy-based transport baseline
#   ./run.sh --size-mib 64       sweep weight size
set -euo pipefail
cd "$(dirname "$0")"

[ -x rl_rollout ] || make

export NVSHMEM_ENABLE_CXL_TRANSPORT=1
export NVSHMEM_CXL_FORCE=1
export NVSHMEM_HEAP_KIND=CXL
export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_DEBUG=WARN
export LD_LIBRARY_PATH="$(cd ../../build/src/lib && pwd):${LD_LIBRARY_PATH:-}"

rm -f /dev/shm/cxl_symm_heap
exec mpirun --allow-run-as-root -np 2 ./rl_rollout "$@"
