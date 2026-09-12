# RL rollout over NVSHMEM-CXL

An RL post-training loop where the policy weights and the generated
trajectories live in the NVSHMEM symmetric heap, and the heap is bound to the
CXL NUMA node.  The rollout worker generates on the RTX 5090 by reading the
weights straight out of CXL over the zero-copy aperture; the trainer updates
the same pages in place and publishes by bumping an 8-byte version.  This is
the `rlcxl` protocol (Splash repo) ported onto NVSHMEM with the CXL transport
and heap added in this fork.

## Roles

    PE 0  trainer (CPU)   consumes trajectories in place from the ring,
                          in-place SGD into the CXL pages, publish = version
                          bump (~60-300 ns measured)
    PE 1  rollout (GPU)   decode loop reads weights over the zero-copy
                          aperture (mode=cxl), pulls one full RDMA copy per
                          new version over verbs (mode=rdma), or stages a
                          cudaMemcpy into VRAM per version the way a
                          conventional transport would (mode=vram); pushes
                          trajectories into a ring in its own heap window

Protocol, unchanged from rlcxl:

| edge | conventional | here |
| --- | --- | --- |
| trainer -> rollout weights | broadcast / IPC / checkpoint, O(model) per step | in-place write into the shared pages, then a version bump: **one 8-bit-store atomic** |
| rollout -> trainer trajectories | serialise, queue, deserialise | ring in the same shared CXL slab, consumed in place |
| rollout weight load | copy into the engine's address space | `nvshmem_ptr` into the peer's window, aliased |

Correctness across the shared weights is a reader-lease protocol, not a lock:
generation runs under a lease that pins the weight version; the trainer drains
the in-flight lease before mutating anything; every trajectory carries the
version it was produced under so the trainer can separate on-policy from
stale samples.

## What was added to the fork to make this work

The upstream fork had a CXL transport and a CXL heap class, but they were
disconnected: the heap class was never instantiated, the transport gated
reachability on NVIDIA RM CXL ioctls (which consumer drivers reject), and
`get_mem_handle` allocated and copied a fresh buffer per registration.

* `src/host/mem/mem_heap_cxl.cpp` — rewritten as a shared-memory slab heap
  (modeled on the upstream sysmem heap): one POSIX shm object per node,
  `mbind(MPOL_BIND)` to the CXL NUMA node **before first touch**,
  `cudaHostRegister` for the zero-copy aperture, plus a `move_pages`
  placement self-check logged at init.  Selected with `NVSHMEM_HEAP_KIND=CXL`.
* `src/host/transport/cxl/cxl.cpp` — `NVSHMEM_CXL_FORCE=1` treats the
  node-local CXL NUMA tier as reachable without RM capability reporting;
  handles are now pure `{address, length}` pairs; rma/amo resolve remote heap
  offsets through the local mapping of the shared slab.
* `NVSHMEMI_HEAP_KIND_CXL_TYPE3` — new heap kind wired through init and the
  SYSMEM-parity gates (team psync host memcpy, LL128 disable, rail-opt).  The
  device-visible `symmetric_heap_kind` stays a bool (the device-state layout
  is size-pinned) and simply records "host-backed"; the exact kind lives in
  the host-only `nvshmemi_host_heap_kind`.

Device-initiated puts need no proxy: the kernel dereferences
`peer_heap_base_p2p[pe]` (the peer's window inside my mapping of the slab) as
a plain volatile store, and the whole slab is device-accessible through the
zero-copy aperture.

## Machine

    AMD Ryzen Threadripper 7960X, 24C/48T, 4 CCXs
    NUMA 0  31 GiB DDR5, all 48 CPUs
    NUMA 1  122 GiB CXL (cxl mem0, PCIe bf:00.0), CPU-less, distance 50
    NVIDIA GeForce RTX 5090 32 GiB (driver 610.57.04, CUDA 13.1 toolkit)
    2 PEs, both on the one GPU (MPG); host-side NVSHMEM atomics and
    collectives are MPG-restricted, so the loop uses mapped heap words only

The placement self-check at init is the proof the heap is where it claims:

    CXL Type 3 heap: slab=0x7fdfc0000000 gpu_base=0x7fdfc0000000
                     my_window=0x7fdfc0000000 per_pe=1073741824 slab=2147483648 node=1
    CXL heap: window placement verified, 64/64 sampled pages on node 1

## Multiple publish vs single copy (RDMA)

`--mode rdma` is the conventional transport done properly: the trainer
registers its weight window as an MR (`REMOTE_READ`) and publishes the QP/rkey
through the ctrl block; on every new version the rollout issues **one** chain
of RDMA READs pulling the whole window into its own registered pages, then
generates from that copy.  The publish plane is unchanged: the version bump
still goes through the shared CXL heap, so the two arms differ *only* in how
new weights reach the rollout.  Transport is SoftRoCE (`rxe0` on `eno1`;
`RDMA_DEVICE` env overrides) because this host's mlx5_0 link is down --
identical structure to hardware RoCE, with the copy cost dominated by the
software loopback instead of the wire.

Sweep (`40_experiments.sh`, 6 rounds x 16 tokens x batch 4; stage = the
per-version single copy; publishes_per_copy = stage_ns / publish_ns):

| mode | weights | gpu ms/round | tok/s | weight read | publish | stage (single copy) | publishes per copy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cxl (multiple publish) | 16 MiB | 11.9 | 1339 | 22.5 GB/s | 60 ns | **none** | -- |
| cxl | 64 MiB | 132.4 | 121 | 8.1 GB/s | 290 ns | **none** | -- |
| cxl | 256 MiB | 754.4 | 21.2 | 5.7 GB/s | 296 ns | **none** | -- |
| rdma (single copy) | 16 MiB | 12.5 | 1282 | 21.6 GB/s | 55 ns | 22.5 ms | 408,182 |
| rdma | 64 MiB | 138.2 | 116 | 7.8 GB/s | 180 ns | 93.8 ms | 521,000 |
| rdma | 256 MiB | 763.4 | 20.9 | 5.6 GB/s | 546 ns | 329.6 ms | 604,271 |
| vram (copy-in, local DMA) | 16 MiB | 0.48 | 35842 | 612 GB/s | 65 ns | 0.68 ms | 10,462 |
| vram | 64 MiB | 1.10 | 15236 | 1037 GB/s | 356 ns | 3.38 ms | 9,508 |
| vram | 256 MiB | 4.89 | 3263 | 879 GB/s | 731 ns | 12.20 ms | 16,689 |

What the comparison isolates: the rdma arm's generation streams from CXL at
the *same* GB/s as the cxl arm (7.8 vs 8.1 at 64 MiB) because both read the
same class of pages -- so the RDMA copy buys nothing here and is pure
per-version overhead.  One RDMA single copy costs as much as **4.1e5-6.0e5
publishes** on SoftRoCE; even the fastest local-DMA copy is worth ~1e4
publishes.  A training run publishes once per step (thousands to millions of
steps), which is why the zero-copy side never sees the weight-sync term at
all.  On hardware RoCE the copy shrinks to ~1-3 ms at 64 MiB (200 Gb/s) --
still ten thousand publishes per copy, and it still leaves the rollout with
a second copy of the weights instead of aliasing the trainer's.



    make -C ../..          # the nvshmem-cxl library
    make                   # this test
    ./run.sh               # cxl mode
    ./run.sh --mode vram   # conventional copy-based baseline
    ./40_experiments.sh    # the size sweep, writes results/*.tsv

## Measured on this host

`40_experiments.sh`, 6 rollout rounds x 16 tokens x batch 4, median-ish over
rounds (results/rl_rollout_{cxl,vram}.tsv):

| mode | weights | gpu ms/round | tok/s | weight read | publish |
| --- | --- | ---: | ---: | ---: | ---: |
| cxl  | 16 MiB | 11.9 | 1348 | 22.6 GB/s | 60 ns |
| cxl  | 64 MiB | 130.0 | 123 | 8.3 GB/s | 60 ns |
| cxl  | 256 MiB | 734.6 | 21.8 | 5.8 GB/s | 186 ns |
| vram | 16 MiB | 0.50 | 35190 | ~600 GB/s | 65 ns |
| vram | 64 MiB | 0.98 | 16522 | ~1120 GB/s | 180 ns |
| vram | 256 MiB | 4.84 | 3303 | ~890 GB/s | 155 ns |

The vram mode also pays `stage_in` on every new version (5.5 ms at 64 MiB,
~10 ms at 256 MiB measured in the run log) that cxl mode never pays, and
vram mode is impossible once the policy stops fitting in the GPU: a 27B fp16
policy is 54 GiB against 32 GiB of VRAM.  The CXL mode has no such bound --
the arena scales with the CXL node (122 GiB here), not the GPU.

Reading the numbers honestly:

* The GPU's zero-copy read stream out of CXL is 5.8-8.3 GB/s when the pages
  are cold (larger models) and ~20 GB/s when the 16 MiB working set is still
  resident in the 32 MiB L3.  For reference, the CPU reads the same tier at
  21.3 GiB/s (`rlcxl results/e1_tiering.tsv`) and the cudaMemcpy smoke test
  sustains 20-22 GB/s at 1 MiB+ transfers -- the single-kernel streaming
  kernel is latency-limited, not link-limited, and there is headroom in
  deeper unrolling / copy-engine use.
* Publish stays O(1) at every model size: 60-186 ns against the 439 ms a
  copy-based transport spends per 1.84 GiB stage-out+apply-in
  (`rlcxl results/e4_sync.tsv`).
* tok/s under CXL streaming (645 at 16 MiB steady state) is ~50x the
  CPU-only rlcxl loop (13.1 tok/s pinned); VRAM-resident weights remain
  25-130x faster than CXL streaming.  The trade is capacity, not speed:
  weights that fit in VRAM should live in VRAM; weights that do not fit
  anywhere local can now participate in the same zero-copy loop.

## Known limitations

* Host-side `nvshmem_*_atomic_*` and collectives abort under MPG (2 PEs, 1
  GPU) -- the loop deliberately uses mapped-heap flags instead.
* The trainer's update is an advantage-weighted outer product on the stashed
  last-layer pair plus weight decay -- a workload stand-in for rlcxl's real
  GRPO update over Qwen2.5-0.5B.  Porting the torch trainer onto the heap
  (torch `frombuffer` over the symmetric heap, as rlcxl's `cxl_arena.py`
  does) is the next step.
* Trajectory pushes are host-side; GPU-initiated pushes through the device
  put path (verified separately in `../cxl_nvshmem_test`) are not wired into
  the ring yet.
* The ring never wraps in this configuration (8 slots, 6 rounds); wraparound
  is implemented but untested under contention.
* The RDMA arm runs over SoftRoCE (`rxe0`); the mlx5_0 NIC on this host is
  link-down.  Hardware RoCE numbers will be 30-100x faster on the copy, which
  moves `publishes_per_copy` down but not the structural result.

## Running it

    make -C ../..          # the nvshmem-cxl library
    make                   # this test
    ./run.sh               # cxl mode (multiple publish)
    ./run.sh --mode rdma   # single RDMA copy per version (SoftRoCE rxe0)
    ./run.sh --mode vram   # local DMA copy-in baseline
    ./40_experiments.sh    # the 3-arm size sweep, writes results/*.tsv

For `--mode rdma`, the SoftRoCE device must exist (one-time):

    modprobe rdma_rxe
    rdma link add rxe0 type rxe netdev eno1

Individual knobs: `--size-mib`, `--layers`, `--batch`, `--tokens`, `--rounds`,
`--steps`, `--lr`; `RDMA_DEVICE=rxe0` selects the verbs device.
