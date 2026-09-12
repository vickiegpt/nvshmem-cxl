/*
 * RL rollout over NVSHMEM-CXL: zero-copy weight streaming from the CXL heap.
 *
 * Port of the rlcxl (Splash) protocol onto NVSHMEM with the CXL symmetric
 * heap (NVSHMEM_HEAP_KIND=CXL: one shared-memory slab mbind'ed to the CXL
 * NUMA node, cudaHostRegister'd for zero-copy GPU access).
 *
 *   PE 0 = trainer (CPU):  consumes trajectories in place, SGD update writes
 *          the weight bytes in the CXL pages, publishes by a version bump.
 *   PE 1 = rollout (GPU):  decode loop reads the weights straight out of the
 *          CXL heap over the zero-copy aperture (mode=cxl) or stages a full
 *          copy into VRAM on every new version like a conventional transport
 *          (mode=vram).  Pushes trajectories into a ring in its own heap
 *          window, consumed in place by the trainer.
 *
 * rlcxl semantics preserved:
 *   - publish is O(1): an 8-byte version bump after the in-place update
 *   - generation runs under a lease pinning the weight version
 *   - the trainer drains the in-flight lease before mutating weights
 *   - every trajectory carries the version it was produced under, so the
 *     trainer can separate on-policy from stale samples
 *
 * Host-side NVSHMEM atomics and collectives are deliberately not used inside
 * the loop: with both PEs on one GPU (MPG) those APIs are restricted, and the
 * mapped-heap words are also what makes the protocol portable to the
 * CXLMemSim environment.
 *
 * Build:  make -C test/rl_rollout
 * Run:    test/rl_rollout/run.sh            (2 PEs, mpirun)
 */

#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <string>
#include <time.h>
#include <utility>

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s:%d %s: %s\n", __FILE__, __LINE__, #call,   \
                    cudaGetErrorString(err_));                                        \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

#define NVSHMEM_CHECK(call)                                                              \
    do {                                                                                 \
        int err_ = (call);                                                               \
        if (err_ != 0) {                                                                 \
            fprintf(stderr, "NVSHMEM error %s:%d %s: %d\n", __FILE__, __LINE__, #call,   \
                    err_);                                                               \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ------------------------------------------------------------------ */
/* Layout (all objects symmetric: every PE nvshmem_mallocs the same    */
/* sizes in the same order, so addresses agree across PEs).            */
/* ------------------------------------------------------------------ */

typedef struct {
    uint64_t weight_version;  /* trainer publishes, rollout reads (acquire) */
    uint64_t lease_active;    /* rollout holds a generation lease           */
    uint64_t ring_tail;       /* rollout-pushed trajectory count            */
    uint64_t ring_head;       /* trainer-consumed trajectory count          */
    uint64_t stop;            /* graceful shutdown flag                     */
    /* RDMA transport exchange (mode=rdma): the trainer publishes its QP and
     * the registered window of the weights; the rollout pulls one copy per
     * version with an RDMA READ instead of aliasing the pages. */
    uint64_t qp_num;
    uint64_t qp_psn;
    uint64_t qp_lid;
    uint64_t qp_mtu;
    uint64_t w_vaddr;
    uint64_t w_rkey;
} ctrl_t;

#define TRAJ_STASH 4096 /* max dim stashed per trajectory, floats */

typedef struct {
    uint64_t version;  /* policy version this trajectory was generated under */
    uint64_t seq;
    float reward;
    float advantage;      /* reward - group baseline */
    uint32_t stash_n;     /* valid length of x/y */
    uint32_t pad;
    float x[TRAJ_STASH];  /* last-layer input snapshot */
    float y[TRAJ_STASH];  /* last-layer output snapshot */
} traj_t;

#define RING_SLOTS 8
#define SPIN_TIMEOUT_NS 30000000000ULL

static inline void spin_sleep(void) {
    struct timespec ts = {0, 200000};
    nanosleep(&ts, NULL);
}

/* ------------------------------------------------------------------ */
/* RDMA single-copy transport (mode=rdma): one verbs RC QP pair over   */
/* SoftRoCE (rxe) by default; the real NIC (mlx5_0) is link-down on    */
/* this host.  Structurally identical to hardware RoCE, with the copy  */
/* cost dominated by the software loopback instead of the wire.        */
/* ------------------------------------------------------------------ */

#define RDMA_CHUNK (16u << 20) /* per-WR RDMA READ size */
#define RDMA_MAX_WR 128
#define RDMA_DEV_ENV "RDMA_DEVICE"

typedef struct {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *w_mr;
} rdma_ctx_t;

static rdma_ctx_t g_rdma;
static uint64_t g_rdma_lid = 0, g_rdma_mtu = 0;
static union ibv_gid g_rdma_gid; /* rxe is Ethernet-linked: AHs are GRH/GID-based */

static inline void cpu_relax(void) { __builtin_ia32_pause(); }

static struct ibv_device *rdma_find_device(const char *want) {
    int nb = 0;
    struct ibv_device **list = ibv_get_device_list(&nb);
    if (!list) return NULL;
    struct ibv_device *found = NULL;
    for (int i = 0; i < nb; i++) {
        if (strcmp(ibv_get_device_name(list[i]), want) == 0) found = list[i];
    }
    return found;
}

static int rdma_setup(const char *devname, void *wbuf, size_t wbytes, int remote_read) {
    memset(&g_rdma, 0, sizeof(g_rdma));
    struct ibv_device *dev = rdma_find_device(devname);
    if (!dev) {
        fprintf(stderr, "RDMA device '%s' not found\n", devname);
        return -1;
    }
    g_rdma.ctx = ibv_open_device(dev);
    g_rdma.pd = ibv_alloc_pd(g_rdma.ctx);
    g_rdma.cq = ibv_create_cq(g_rdma.ctx, RDMA_MAX_WR + 16, NULL, NULL, 0);
    if (!g_rdma.ctx || !g_rdma.pd || !g_rdma.cq) {
        fprintf(stderr, "verbs alloc failed (ctx=%p pd=%p cq=%p)\n", g_rdma.ctx, g_rdma.pd,
                g_rdma.cq);
        return -1;
    }

    struct ibv_qp_init_attr qia = {};
    qia.send_cq = g_rdma.cq;
    qia.recv_cq = g_rdma.cq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr = RDMA_MAX_WR;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_wr = 1;
    g_rdma.qp = ibv_create_qp(g_rdma.pd, &qia);
    if (!g_rdma.qp) {
        fprintf(stderr, "ibv_create_qp failed errno=%d\n", errno);
        return -1;
    }

    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.qp_access_flags =
        IBV_ACCESS_LOCAL_WRITE | (remote_read ? IBV_ACCESS_REMOTE_READ : 0);
    attr.port_num = 1;
    if (ibv_modify_qp(g_rdma.qp, &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        fprintf(stderr, "IBV_QPS_INIT failed\n");
        return -1;
    }

    /* Register the weight window.  The trainer exposes it REMOTE_READ; the
     * rollout's window is the RDMA READ destination. */
    int acc = IBV_ACCESS_LOCAL_WRITE | (remote_read ? IBV_ACCESS_REMOTE_READ : 0);
    g_rdma.w_mr = ibv_reg_mr(g_rdma.pd, wbuf, wbytes, (ibv_access_flags)acc);
    if (!g_rdma.w_mr) {
        fprintf(stderr, "ibv_reg_mr(%zu bytes) failed errno=%d\n", wbytes, errno);
        return -1;
    }

    struct ibv_port_attr pattr = {};
    if (ibv_query_port(g_rdma.ctx, 1, &pattr)) {
        fprintf(stderr, "ibv_query_port failed\n");
        return -1;
    }
    g_rdma_lid = pattr.lid;
    g_rdma_mtu = (uint64_t)pattr.active_mtu;
    if (ibv_query_gid(g_rdma.ctx, 1, 0, &g_rdma_gid)) {
        fprintf(stderr, "ibv_query_gid failed\n");
        return -1;
    }
    return 0;
}

static int rdma_connect_rtr_rts(uint64_t peer_qpn, uint64_t peer_psn, uint64_t peer_lid,
                                uint64_t mtu, uint64_t my_psn) {
    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = (enum ibv_mtu)mtu;
    attr.dest_qp_num = (uint32_t)peer_qpn;
    attr.rq_psn = (uint32_t)peer_psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    /* rxe presents link-layer Ethernet: address the peer by GID (GRH), not LID */
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = g_rdma_gid; /* loopback: both PEs share the port GID */
    attr.ah_attr.grh.sgid_index = 0;
    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.grh.traffic_class = 0;
    attr.ah_attr.dlid = (uint16_t)peer_lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = 1;
    if (ibv_modify_qp(g_rdma.qp, &attr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
        fprintf(stderr, "IBV_QPS_RTR failed\n");
        return -1;
    }

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 3;
    attr.rnr_retry = 3;
    attr.sq_psn = (uint32_t)my_psn;
    attr.max_rd_atomic = 1;
    if (ibv_modify_qp(g_rdma.qp, &attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
        fprintf(stderr, "IBV_QPS_RTS failed\n");
        return -1;
    }
    return 0;
}

/* The single copy: one chained set of RDMA READs pulling the whole weight
 * window from the trainer's registered pages.  Returns elapsed ns. */
static uint64_t rdma_pull_weights(uint64_t remote_vaddr, uint64_t remote_rkey, void *dst,
                                  size_t wbytes) {
    uint64_t t0 = now_ns();
    size_t nchunks = (wbytes + RDMA_CHUNK - 1) / RDMA_CHUNK;
    struct ibv_send_wr *bad = NULL;
    struct ibv_wc wc;

    for (size_t i = 0; i < nchunks; i++) {
        size_t off = i * RDMA_CHUNK;
        size_t len = wbytes - off < RDMA_CHUNK ? wbytes - off : RDMA_CHUNK;
        struct ibv_sge sge = {};
        sge.addr = (uintptr_t)dst + off;
        sge.length = (uint32_t)len;
        sge.lkey = g_rdma.w_mr->lkey;

        struct ibv_send_wr wr = {};
        wr.wr_id = i;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_READ;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = remote_vaddr + off;
        wr.wr.rdma.rkey = (uint32_t)remote_rkey;
        if (ibv_post_send(g_rdma.qp, &wr, &bad)) {
            fprintf(stderr, "ibv_post_send failed\n");
            return ~0ull;
        }
        /* pipelined postings; drain one completion per signal */
        int n;
        while ((n = ibv_poll_cq(g_rdma.cq, 1, &wc)) == 0) cpu_relax();
        if (n < 0 || wc.status != IBV_WC_SUCCESS) {
            fprintf(stderr, "RDMA READ failed: status=%d (%s)\n", wc.status,
                    ibv_wc_status_str(wc.status));
            return ~0ull;
        }
    }
    return now_ns() - t0;
}

/* ------------------------------------------------------------------ */
/* Model shim: L layers of n x n fp32 weights; a decode step is        */
/* x = relu(W_l @ x) per layer.  Weights live in the CXL heap.         */
/* ------------------------------------------------------------------ */

/* y[b, :] = relu(W @ x[b, :]); warp-per-row float4 streaming GEMV.
 * In cxl mode W is a host (CXL) pointer the GPU dereferences through the
 * zero-copy aperture; in vram mode it is a plain device pointer.  Either
 * way the whole layer is streamed once per token. */
__global__ void gemv_layer(const float *__restrict__ W, const float *__restrict__ x,
                           float *__restrict__ y, int n, int batch) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (row >= n) return;

    const float4 *w4 = (const float4 *)(W + (size_t)row * n);
    int n4 = n / 4;

    for (int b = 0; b < batch; b++) {
        float acc = 0.f;
        const float *xb = x + (size_t)b * n;
        /* two independent loads in flight per lane: doubles outstanding
         * reads so the CXL link, not warp latency, bounds the stream */
        int j = lane;
        for (; j + 32 < n4; j += 64) {
            float4 w0 = w4[j];
            float4 w1 = w4[j + 32];
            acc += w0.x * xb[4 * j + 0] + w0.y * xb[4 * j + 1] + w0.z * xb[4 * j + 2] +
                   w0.w * xb[4 * j + 3];
            acc += w1.x * xb[4 * (j + 32) + 0] + w1.y * xb[4 * (j + 32) + 1] +
                   w1.z * xb[4 * (j + 32) + 2] + w1.w * xb[4 * (j + 32) + 3];
        }
        for (; j < n4; j += 32) {
            float4 w = w4[j];
            acc += w.x * xb[4 * j + 0] + w.y * xb[4 * j + 1] + w.z * xb[4 * j + 2] +
                   w.w * xb[4 * j + 3];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) y[(size_t)b * n + row] = fmaxf(acc, 0.f);
    }
}

typedef struct {
    int layers = 4;
    int dim = 1024;
    int batch = 4;
    int tokens = 16;
    int rounds = 6;
    int steps = 2;            /* trainer updates; one per round at most */
    float lr = 0.01f;
    long size_mib = 0;        /* override total weight bytes */
    std::string mode = "cxl"; /* cxl | vram */
} opts_t;

static opts_t g_opts;

static size_t compute_weight_bytes(void) {
    if (g_opts.size_mib > 0) {
        size_t per = ((size_t)(g_opts.size_mib << 20)) / g_opts.layers / sizeof(float);
        return per * g_opts.layers * sizeof(float);
    }
    return (size_t)g_opts.layers * g_opts.dim * g_opts.dim * sizeof(float);
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        auto arg = std::string(argv[i]);
        auto val = [&]() -> const char * {
            if (i + 1 < argc) return argv[++i];
            fprintf(stderr, "missing value for %s\n", argv[i]);
            exit(2);
        };
        if (arg == "--layers") g_opts.layers = atoi(val());
        else if (arg == "--dim") g_opts.dim = atoi(val());
        else if (arg == "--batch") g_opts.batch = atoi(val());
        else if (arg == "--tokens") g_opts.tokens = atoi(val());
        else if (arg == "--rounds") g_opts.rounds = atoi(val());
        else if (arg == "--steps") g_opts.steps = atoi(val());
        else if (arg == "--lr") g_opts.lr = atof(val());
        else if (arg == "--size-mib") g_opts.size_mib = atol(val());
        else if (arg == "--mode") g_opts.mode = val();
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); exit(2); }
    }
    if (g_opts.mode != "cxl" && g_opts.mode != "vram" && g_opts.mode != "rdma") {
        fprintf(stderr, "--mode must be cxl, vram, or rdma\n");
        return 2;
    }

    nvshmem_init(); /* exits on failure */
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    if (npes != 2) {
        fprintf(stderr, "this program needs exactly 2 PEs (trainer=0, rollout=1)\n");
        return 2;
    }
    CUDA_CHECK(cudaSetDevice(0));

    size_t wbytes = compute_weight_bytes();
    int n = (int)sqrtf((float)(wbytes / g_opts.layers / sizeof(float)));
    n &= ~3;
    if (n > TRAJ_STASH) {
        if (mype == 0)
            fprintf(stderr, "dim %d exceeds trajectory stash %d; lower --size-mib/--dim\n", n,
                    TRAJ_STASH);
        return 2;
    }
    wbytes = (size_t)g_opts.layers * n * n * sizeof(float);

    if (mype == 0) {
        printf("[rl-rollout] mode=%s layers=%d dim=%d batch=%d tokens=%d rounds=%d steps=%d "
               "weights=%.1f MiB\n",
               g_opts.mode.c_str(), g_opts.layers, n, g_opts.batch, g_opts.tokens,
               g_opts.rounds, g_opts.steps, (double)wbytes / 1048576.0);
    }

    /* -------- symmetric allocations (same order, same sizes) -------- */
    ctrl_t *ctrl = (ctrl_t *)nvshmem_malloc(sizeof(ctrl_t));
    float *W_sym = (float *)nvshmem_malloc(wbytes);
    traj_t *ring = (traj_t *)nvshmem_malloc(sizeof(traj_t) * RING_SLOTS);
    if (!ctrl || !W_sym || !ring) {
        fprintf(stderr, "[PE %d] nvshmem_malloc failed\n", mype);
        return 1;
    }

    /* peer objects through my own mapping of the shared CXL slab */
    ctrl_t *peer_ctrl = (ctrl_t *)nvshmem_ptr(ctrl, mype ^ 1);
    float *peer_W = (float *)nvshmem_ptr(W_sym, mype ^ 1);
    traj_t *peer_ring = (traj_t *)nvshmem_ptr(ring, mype ^ 1);

    CUDA_CHECK(cudaMemset(ctrl, 0, sizeof(ctrl_t)));
    CUDA_CHECK(cudaMemset(ring, 0, sizeof(traj_t) * RING_SLOTS));
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    /* -------- RDMA transport (mode=rdma): exchange QP/MR via the ctrl block */
    if (g_opts.mode == "rdma") {
        const char *devname = getenv(RDMA_DEV_ENV);
        if (!devname || !*devname) devname = "rxe0";
        if (rdma_setup(devname, W_sym, wbytes, /*remote_read=*/mype == 0)) return 1;

        /* my QP + (trainer only) the registered weight window */
        ctrl->qp_num = (uint64_t)g_rdma.qp->qp_num;
        ctrl->qp_psn = 0;
        ctrl->qp_lid = g_rdma_lid;
        ctrl->qp_mtu = g_rdma_mtu;
        if (mype == 0) {
            ctrl->w_vaddr = (uint64_t)(uintptr_t)W_sym;
            ctrl->w_rkey = (uint64_t)g_rdma.w_mr->rkey;
        }
        if (mype == 0)
            printf("[rdma] device QP %u lid %u mtu %llu, weights registered at %p rkey %x\n",
                   (unsigned)g_rdma.qp->qp_num, (unsigned)g_rdma_lid,
                   (unsigned long long)g_rdma_mtu, W_sym, g_rdma.w_mr->rkey);
        /* both sides wait for the peer's QP before transitioning to RTS */
        uint64_t q0 = now_ns();
        while (__atomic_load_n(&peer_ctrl->qp_num, __ATOMIC_ACQUIRE) == 0) {
            spin_sleep();
            if (now_ns() - q0 > SPIN_TIMEOUT_NS) {
                fprintf(stderr, "[PE %d] timeout waiting for peer QP\n", mype);
                return 1;
            }
        }
        if (rdma_connect_rtr_rts(peer_ctrl->qp_num, peer_ctrl->qp_psn, peer_ctrl->qp_lid,
                                 peer_ctrl->qp_mtu, /*my_psn=*/0))
            return 1;
        nvshmem_barrier_all();
    }

    /* ------------------------------- trainer ------------------------------- */
    if (mype == 0) {
        float *W = (float *)malloc(wbytes);
        size_t nn = (size_t)n * n;
        for (size_t i = 0; i < nn * (size_t)g_opts.layers; i++)
            W[i] = 0.01f * (float)((i * 2654435761u) % 97) / 97.f - 0.005f;
        memcpy(W_sym, W, wbytes);
        __atomic_store_n(&ctrl->weight_version, (uint64_t)1, __ATOMIC_RELEASE);
        printf("[trainer] v1 published (%.1f MiB in CXL heap)\n", wbytes / 1048576.0);

        uint64_t consumed = 0;
        double drain_total_ms = 0, opt_total_ms = 0;
        uint64_t publish_ns_min = ~0ull, publish_ns_max = 0, publish_ns_sum = 0;
        uint64_t stale_total = 0, ring_full_total = 0;
        int step = 0;

        for (int round = 0; round < g_opts.rounds && step < g_opts.steps; round++) {
            /* drain: wait for the in-flight lease so the update is on-policy */
            uint64_t t0 = now_ns();
            while (__atomic_load_n(&peer_ctrl->lease_active, __ATOMIC_ACQUIRE) != 0) {
                spin_sleep();
                if (now_ns() - t0 > SPIN_TIMEOUT_NS) {
                    fprintf(stderr, "[trainer] timeout draining lease\n");
                    return 1;
                }
            }
            uint64_t drain_ns = now_ns() - t0;
            drain_total_ms += drain_ns / 1e6;

            /* wait for a trajectory to consume (the first step blocks here) */
            uint64_t t0b = now_ns();
            while (__atomic_load_n(&peer_ctrl->ring_tail, __ATOMIC_ACQUIRE) <= consumed &&
                   !__atomic_load_n(&ctrl->stop, __ATOMIC_RELAXED)) {
                spin_sleep();
                if (now_ns() - t0b > SPIN_TIMEOUT_NS) {
                    fprintf(stderr, "[trainer] timeout waiting for trajectories\n");
                    return 1;
                }
            }

            /* collect: consume ring slots in place */
            uint64_t tail = __atomic_load_n(&peer_ctrl->ring_tail, __ATOMIC_ACQUIRE);
            double reward_sum = 0;
            uint64_t produced = tail - consumed;
            uint64_t cur_v = __atomic_load_n(&ctrl->weight_version, __ATOMIC_RELAXED);
            for (uint64_t s = consumed; s < tail; s++) {
                traj_t *t = &peer_ring[s % RING_SLOTS];
                reward_sum += t->reward;
                if (t->version != cur_v) stale_total++;
            }
            consumed = tail;
            /* free the slots we just consumed */
            __atomic_store_n(&ctrl->ring_head, tail, __ATOMIC_RELEASE);

            /* opt: in-place SGD into the CXL pages, shaped by the consumed
             * trajectories: advantage-weighted outer product on the stashed
             * last-layer pair plus weight decay.  A workload stand-in for the
             * GRPO update in rlcxl/trainer.py. */
            uint64_t t1 = now_ns();
            if (produced > 0) {
                float adv = (float)(reward_sum / (double)produced) - 0.5f;
                for (uint64_t s = tail - produced; s < tail; s++) {
                    traj_t *t = &peer_ring[s % RING_SLOTS];
                    int sn = (int)t->stash_n;
                    for (int i = 0; i < sn; i++) {
                        float g = adv * t->y[i] * t->x[i];
                        W_sym[i] -= g_opts.lr * g;               /* layer 0 head */
                        W_sym[nn + i] -= g_opts.lr * 0.01f * g;  /* layer 1 echo */
                    }
                }
                for (size_t i = 0; i < nn * (size_t)g_opts.layers; i++)
                    W_sym[i] -= g_opts.lr * 1e-5f * W_sym[i];
            }
            uint64_t opt_ns = now_ns() - t1;
            opt_total_ms += opt_ns / 1e6;

            /* commit: the O(1) publish */
            uint64_t t2 = now_ns();
            uint64_t v = __atomic_add_fetch(&ctrl->weight_version, 1, __ATOMIC_RELEASE);
            uint64_t publish_ns = now_ns() - t2;
            publish_ns_min = publish_ns < publish_ns_min ? publish_ns : publish_ns_min;
            publish_ns_max = publish_ns > publish_ns_max ? publish_ns : publish_ns_max;
            publish_ns_sum += publish_ns;
            step++;
            printf("[trainer] step %d: n=%llu reward=%.4f stale=%llu drain=%.1fms opt=%.1fms "
                   "publish=%lluns -> v%llu\n",
                   step - 1, (unsigned long long)produced, reward_sum / (double)(produced ?: 1),
                   (unsigned long long)stale_total, drain_ns / 1e6, opt_ns / 1e6,
                   (unsigned long long)publish_ns, (unsigned long long)v);
        }

        __atomic_store_n(&ctrl->stop, (uint64_t)1, __ATOMIC_RELEASE);
        printf("[trainer] done: drain=%.1fms opt=%.1fms publish(min/avg/max)=%llu/%llu/%llu ns "
               "stale=%llu ring_full=%llu\n",
               drain_total_ms, opt_total_ms, (unsigned long long)publish_ns_min,
               (unsigned long long)(step ? publish_ns_sum / step : 0),
               (unsigned long long)publish_ns_max, (unsigned long long)stale_total,
               (unsigned long long)ring_full_total);
        free(W);
    }

    /* ------------------------------- rollout ------------------------------- */
    if (mype == 1) {
        float *d_x, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * g_opts.batch * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, (size_t)n * g_opts.batch * sizeof(float)));
        float *d_W_vram = NULL;
        if (g_opts.mode == "vram") CUDA_CHECK(cudaMalloc(&d_W_vram, wbytes));
        /* generation source per mode: alias the trainer's pages (cxl), the
         * RDMA-pulled local window (rdma), or the staged VRAM copy (vram) */
        const float *W_src = (g_opts.mode == "cxl") ? peer_W
                             : (g_opts.mode == "rdma") ? W_sym
                                                       : d_W_vram;

        /* deterministic "prompts": per-sequence constant input */
        float *h_x = (float *)malloc((size_t)n * g_opts.batch * sizeof(float));
        for (int b = 0; b < g_opts.batch; b++)
            for (int i = 0; i < n; i++) h_x[(size_t)b * n + i] = ((i + b) % 16) / 16.f;
        CUDA_CHECK(
            cudaMemcpy(d_x, h_x, (size_t)n * g_opts.batch * sizeof(float), cudaMemcpyHostToDevice));

        /* reward target: a fixed direction; reward = normalized overlap of the
         * last-layer output with it, centered on 0.5 */
        float *target = (float *)malloc((size_t)n * sizeof(float));
        for (int i = 0; i < n; i++) target[i] = ((i * 7) % 23) / 23.f - 0.5f;
        float *h_y = (float *)malloc((size_t)n * g_opts.batch * sizeof(float));

        uint64_t staged_version = 0;
        double stage_total_ms = 0;
        double last_stage_ms = 0;
        uint64_t pushed = 0, ring_full = 0;

        int threads = 256;
        int blocks = (n * 32 + threads - 1) / threads;

        for (int round = 0; round < g_opts.rounds; round++) {
            if (__atomic_load_n(&peer_ctrl->stop, __ATOMIC_ACQUIRE)) break;

            /* wait for the first real publish before the first lease */
            uint64_t v;
            uint64_t wait0 = now_ns();
            do {
                v = __atomic_load_n(&peer_ctrl->weight_version, __ATOMIC_ACQUIRE);
                if (v == 0) {
                    spin_sleep();
                    if (now_ns() - wait0 > SPIN_TIMEOUT_NS) {
                        fprintf(stderr, "[rollout] timeout waiting for v1\n");
                        break;
                    }
                }
            } while (v == 0);
            __atomic_store_n(&ctrl->lease_active, (uint64_t)1, __ATOMIC_RELEASE);

            /* conventional transports pay O(model) here; CXL pays nothing */
            last_stage_ms = 0;
            if (g_opts.mode == "vram" && v != staged_version) {
                uint64_t s0 = now_ns();
                CUDA_CHECK(cudaMemcpy(d_W_vram, peer_W, wbytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaDeviceSynchronize());
                last_stage_ms = (now_ns() - s0) / 1e6;
                stage_total_ms += last_stage_ms;
                staged_version = v;
            } else if (g_opts.mode == "rdma" && v != staged_version) {
                /* the single copy: one RDMA READ of the whole window */
                uint64_t pull_ns =
                    rdma_pull_weights(peer_ctrl->w_vaddr, peer_ctrl->w_rkey, W_sym, wbytes);
                if (pull_ns == ~0ull) return 1;
                last_stage_ms = pull_ns / 1e6;
                stage_total_ms += last_stage_ms;
                staged_version = v;
            }

            cudaEvent_t ev0, ev1;
            CUDA_CHECK(cudaEventCreate(&ev0));
            CUDA_CHECK(cudaEventCreate(&ev1));
            uint64_t g0 = now_ns();
            CUDA_CHECK(cudaEventRecord(ev0));
            for (int tok = 0; tok < g_opts.tokens; tok++) {
                for (int l = 0; l < g_opts.layers; l++) {
                    const float *Wl = W_src + (size_t)l * n * n;
                    gemv_layer<<<blocks, threads>>>(Wl, d_x, d_y, n, g_opts.batch);
                    CUDA_CHECK(cudaGetLastError());
                    std::swap(d_x, d_y);
                }
            }
            CUDA_CHECK(cudaEventRecord(ev1));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            float gpu_ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev0, ev1));
            uint64_t round_ns = now_ns() - g0;
            CUDA_CHECK(cudaEventDestroy(ev0));
            CUDA_CHECK(cudaEventDestroy(ev1));

            /* reward from the last layer output (d_x points at the freshest
             * buffer after the swaps) */
            float *out_dev = (g_opts.layers % 2 == 0) ? d_x : d_y;
            CUDA_CHECK(cudaMemcpy(h_y, out_dev, (size_t)n * g_opts.batch * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            float group_reward = 0;
            for (int b = 0; b < g_opts.batch; b++) {
                float dot = 0, ny = 0, nx = 0;
                for (int i = 0; i < n; i++) {
                    float yv = h_y[(size_t)b * n + i];
                    dot += yv * target[i];
                    ny += yv * yv;
                    nx += target[i] * target[i];
                }
                group_reward += 0.5f + dot / (sqrtf(ny * nx) + 1e-9f);
            }
            group_reward /= g_opts.batch;

            /* push into the ring in MY window; the trainer consumes it in
             * place through its own mapping of the same slab */
            uint64_t tail = __atomic_load_n(&ctrl->ring_tail, __ATOMIC_RELAXED);
            uint64_t head = __atomic_load_n(&peer_ctrl->ring_head, __ATOMIC_RELAXED);
            if (tail - head >= RING_SLOTS) {
                ring_full++;
            } else {
                traj_t *t = &ring[tail % RING_SLOTS];
                t->version = v;
                t->seq = tail;
                t->reward = group_reward;
                t->advantage = group_reward - 0.5f;
                t->stash_n = (uint32_t)n;
                memcpy(t->x, h_x, (size_t)n * sizeof(float));
                memcpy(t->y, h_y, (size_t)n * sizeof(float));
                __atomic_store_n(&ctrl->ring_tail, tail + 1, __ATOMIC_RELEASE);
                pushed++;
            }

            double wgb = (double)wbytes * g_opts.tokens / 1e9;
            printf("[rollout] round %d: v=%llu %s gpu=%.2fms wall=%.2fms tok/s=%.1f "
                   "weight-read=%.1fGB/s stage=%.2fms reward=%.4f\n",
                   round, (unsigned long long)v, g_opts.mode.c_str(), gpu_ms, round_ns / 1e6,
                   g_opts.tokens * 1e3 / (round_ns / 1e6), wgb / (gpu_ms / 1e3), last_stage_ms,
                   group_reward);

            __atomic_store_n(&ctrl->lease_active, (uint64_t)0, __ATOMIC_RELEASE);

            /* on-policy cadence: wait until the trainer publishes the next
             * version (or shuts down) before leasing again */
            uint64_t w0 = now_ns();
            while (__atomic_load_n(&peer_ctrl->weight_version, __ATOMIC_ACQUIRE) == v &&
                   !__atomic_load_n(&peer_ctrl->stop, __ATOMIC_RELAXED)) {
                spin_sleep();
                if (now_ns() - w0 > SPIN_TIMEOUT_NS) {
                    fprintf(stderr, "[rollout] timeout waiting for next version\n");
                    break;
                }
            }
        }

        __atomic_store_n(&ctrl->lease_active, (uint64_t)0, __ATOMIC_RELEASE);
        printf("[rollout] done: pushed=%llu ring_full=%llu stage_in=%.1fms total\n",
               (unsigned long long)pushed, (unsigned long long)ring_full, stage_total_ms);
        free(h_x);
        free(h_y);
        free(target);
        if (d_W_vram) CUDA_CHECK(cudaFree(d_W_vram));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
    }

    nvshmem_barrier_all();
    nvshmem_finalize();
    return 0;
}
