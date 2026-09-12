/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 *
 * CXL Type 3 Symmetric Heap Implementation
 *
 * One POSIX shared-memory slab per node, carved into per-PE windows.  The slab
 * is bound with mbind(MPOL_BIND) to the CXL NUMA node (the CPU-less memory
 * expander tier) *before* first touch, so every page of the symmetric heap
 * physically lives in CXL memory.  The slab is then cudaHostRegister'd so the
 * GPU reaches it through the zero-copy aperture, which is the data path for
 * device-initiated NVSHMEM operations (plain stores through
 * peer_heap_base_p2p, no proxy in the data path).
 *
 * Modeled on nvshmemi_symmetric_heap_sysmem_static_shm; the differences are
 * the NUMA binding, the placement self-check, and the slab-offset translation
 * helpers the CXL transport uses.
 */

#include <cuda_runtime.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <vector>

#include "bootstrap_host_transport/env_defs_internal.h"
#include "internal/host/debug.h"
#include "internal/host/nvshmem_internal.h"
#include "internal/host/nvshmemi_symmetric_heap.hpp"
#include "internal/host/nvshmemi_types.h"
#include "internal/host/shared_memory.h"
#include "internal/host/util.h"
#include "internal/host_transport/cudawrap.h"

#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif
#ifndef MPOL_MF_MOVE
#define MPOL_MF_MOVE 2
#endif

#define MEM_HANDLE_DATA(h) ((uint64_t *)((h)->reserved))

static inline long cxl_sys_mbind(void *addr, unsigned long len, int mode,
                                 const unsigned long *nmask, unsigned long maxnode,
                                 unsigned flags) {
    return syscall(__NR_mbind, addr, len, mode, nmask, maxnode, flags);
}

static inline long cxl_sys_move_pages(int pid, unsigned long count, void **pages,
                                      const int *nodes, int *status, int flags) {
    return syscall(__NR_move_pages, pid, count, pages, nodes, status, flags);
}

/* A CPU-less NUMA node is the CXL memory-expander tier on this class of host. */
static int cxl_autodetect_numa_node(void) {
    DIR *dir = opendir("/sys/devices/system/node");
    if (!dir) return -1;

    struct dirent *ent;
    int best = -1;
    while ((ent = readdir(dir)) != NULL) {
        int node;
        if (sscanf(ent->d_name, "node%d", &node) != 1) continue;

        char path[256];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
        FILE *f = fopen(path, "r");
        if (!f) return node; /* no cpulist at all => no CPUs */
        char buf[256] = {0};
        size_t n = fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f);
        bool empty = true;
        for (size_t i = 0; i < n; i++) {
            if (buf[i] != ' ' && buf[i] != '\t' && buf[i] != '\n' && buf[i] != '\r' &&
                buf[i] != '\0') {
                empty = false;
                break;
            }
        }
        if (empty) {
            best = node;
            break;
        }
    }
    closedir(dir);
    return best;
}

int cxl_heap_numa_node(void) {
    int configured = nvshmemi_options.CXL_NUMA_NODE;
    if (configured == -2) return -1;      /* binding explicitly disabled */
    if (configured >= 0) return configured;
    return cxl_autodetect_numa_node();    /* -1 autodetect */
}

/* Bind an existing mapping to a NUMA node before first touch. */
static int cxl_bind_numa(void *addr, size_t len, int node) {
    if (node < 0) return 0;
    unsigned long nmask = 1UL << (node % (8 * sizeof(unsigned long)));
    if (cxl_sys_mbind(addr, len, MPOL_BIND, &nmask, sizeof(nmask) * 8, MPOL_MF_MOVE) != 0) {
        INFO(NVSHMEM_MEM, "CXL heap: mbind(node %d) failed: %s\n", node, strerror(errno));
        return -1;
    }
    return 0;
}

/* Sample pages with move_pages and report how many landed on `node`. */
static void cxl_verify_placement(const char *what, void *addr, size_t len, int node) {
    if (node < 0) return;
    enum { MAX_SAMPLES = 64 };
    size_t page = 4096;
    size_t npages = len / page;
    unsigned long count = npages < MAX_SAMPLES ? (unsigned long)npages : MAX_SAMPLES;
    if (count == 0) return;

    void *pages[MAX_SAMPLES];
    int status[MAX_SAMPLES] = {0};
    for (unsigned long i = 0; i < count; i++) {
        pages[i] = (char *)addr + (i * npages / count) * page;
    }

    if (cxl_sys_move_pages(0, count, pages, NULL, status, 0) != 0) {
        INFO(NVSHMEM_MEM, "CXL heap: move_pages check failed: %s\n", strerror(errno));
        return;
    }

    unsigned on_node = 0;
    for (unsigned long i = 0; i < count; i++) {
        if (status[i] == node) on_node++;
    }
    if (on_node == count) {
        INFO(NVSHMEM_INIT, "CXL heap: %s placement verified, %lu/%lu sampled pages on node %d\n",
             what, on_node, count, node);
    } else {
        WARN("CXL heap: %s placement OFF TARGET, only %lu/%lu sampled pages on node %d\n", what,
             on_node, count, node);
    }
}

/*
 * Shared with the CXL transport (cxl.cpp): where the slab lives in this
 * process, how big one PE window is, and which NUMA node it was bound to.
 */
static void *cxl_heap_slab_base_ = nullptr;
static size_t cxl_heap_window_size_ = 0;
static int cxl_heap_node_ = -1;

extern "C" void *nvshmemi_cxl_heap_slab_base(void) { return cxl_heap_slab_base_; }
extern "C" size_t nvshmemi_cxl_heap_window_size(void) { return cxl_heap_window_size_; }
extern "C" int nvshmemi_cxl_heap_numa_node(void) { return cxl_heap_node_; }

class nvshmemi_symmetric_heap_cxl_type3 : public nvshmemi_symmetric_heap_static {
   public:
    explicit nvshmemi_symmetric_heap_cxl_type3(nvshmemi_state_t *state) noexcept
        : nvshmemi_symmetric_heap_static(state) {}

    ~nvshmemi_symmetric_heap_cxl_type3() { cleanup_symmetric_heap(); }

    static void atexit_heap_handler(void) {
        for (auto i = 0U; i < infos_.size(); i++) {
            INFO(NVSHMEM_MEM, "Closing file descriptor: %d for CXL sym heap\n", infos_[i].shm_fd);
            close(infos_[i].shm_fd);
        }
    }

    int cleanup_symmetric_heap(void) override {
        cleanup_mspace();
        if (heap_base_) {
            free_heap_memory(heap_base_);
            heap_base_ = nullptr;
        }
        return 0;
    }

   protected:
    /* Mirrors nvshmemi_symmetric_heap_static::reserve_heap (sizes, allocate,
     * SYNC_MEMOPS, mspace) but derives the per-PE window from
     * NVSHMEM_CXL_HEAP_SIZE instead of NVSHMEM_SYMMETRIC_SIZE. */
    int reserve_heap(void) override {
        int status;
        size_t heapextra = 0, alignbytes = 0;
        mem_granularity_ = nvshmemi_options.CUMEM_GRANULARITY < NVSHMEMI_MAX_HANDLE_LENGTH
                               ? nvshmemi_options.CUMEM_GRANULARITY
                               : NVSHMEMI_MAX_HANDLE_LENGTH;
        set_heap_size_attr(mem_granularity_, &heapextra, &alignbytes, &(log2_mem_granularity_));

        size_t per_pe = nvshmemi_options.CXL_HEAP_SIZE;
        if (per_pe == 0) per_pe = 256 * 1024 * 1024;
        heap_size_ = NVSHMEMU_ROUND_UP(per_pe + heapextra, mem_granularity_);
        physical_internal_heap_size_ = 0;

        bool data =
            true; /*A boolean attribute which when set, ensures that synchronous memory operations
                    initiated on the region of memory that ptr points to will always synchronize.*/

        status = allocate_heap_memory();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                              "CXL heap allocation failed \n");

        status = CUPFN(
            nvshmemi_cuda_syms,
            cuPointerSetAttribute(&data, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)(heap_base_)));
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                              "cuPointerSetAttribute failed \n");

        INFO(NVSHMEM_MEM,
             "CXL Type 3 heap: base %p per-PE window %zu (NVSHMEM_CXL_HEAP_SIZE %zu + extra %zu), "
             "node %d\n",
             heap_base_, heap_size_, per_pe, heapextra, cxl_heap_node_);

        status = setup_mspace();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "memory space initialization failed \n");

    out:
        if (status) {
            free_heap_memory(heap_base_);
        }

        return (status);
    }

    int allocate_heap_memory() override {
        int status = 0;
        size_t per_pe = heap_size_;
        size_t shm_size = per_pe * nvshmemi_boot_handle.npes_node;

        int ret = snprintf(heap_name_, sizeof(heap_name_), "cxl_symm_heap");
        if (ret < 0) NVSHMEMI_ERROR_EXIT("snprintf failed\n");

        if (nvshmemi_boot_handle.mype_node == 0) {
            if (shared_memory_create(heap_name_, shm_size, &heap_info_) != 0) {
                NVSHMEMI_ERROR_EXIT("Failed to create CXL shared memory slab\n");
            }
        }

        status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
        if (nvshmemi_boot_handle.mype_node != 0) {
            if (shared_memory_open(heap_name_, shm_size, &heap_info_) != 0) {
                NVSHMEMI_ERROR_EXIT("Failed to open CXL shared memory slab\n");
            }
        }

        atexit(nvshmemi_symmetric_heap_cxl_type3::atexit_heap_handler);
        nvshmemi_symmetric_heap_cxl_type3::infos_.push_back(heap_info_);

        /* Bind before any first touch so every page allocates on the CXL node.
         * mbind policy is per-process, so every PE applies it to its own mapping. */
        cxl_heap_node_ = cxl_heap_numa_node();
        cxl_bind_numa(heap_info_.addr, shm_size, cxl_heap_node_);

        status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);

        /* First touch: each PE claims its own window on the bound node. */
        memset((char *)heap_info_.addr + nvshmemi_boot_handle.mype_node * per_pe, 0, per_pe);

        status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);

        CUDA_RUNTIME_CHECK(cudaHostRegister(heap_info_.addr, shm_size, cudaHostRegisterDefault));
        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&global_heap_base_, heap_info_.addr, 0));

        heap_base_ = (char *)global_heap_base_ + nvshmemi_boot_handle.mype_node * per_pe;
        reserved_heap_size_ = shm_size;

        cxl_heap_slab_base_ = heap_info_.addr;
        cxl_heap_window_size_ = per_pe;

        INFO(NVSHMEM_INIT,
             "CXL Type 3 heap: slab=%p gpu_base=%p my_window=%p per_pe=%zu slab=%zu node=%d\n",
             heap_info_.addr, global_heap_base_, heap_base_, per_pe, shm_size, cxl_heap_node_);

        cxl_verify_placement("window", heap_base_, per_pe, cxl_heap_node_);

        return status;
    }

    int free_heap_memory(void *addr __attribute__((unused))) override {
        if (cxl_heap_slab_base_ == nullptr) return 0;

        /* Teardown can race with CUDA runtime unload; the mapping dies with
         * the process either way, so never exit() from here. */
        cudaError_t err = cudaHostUnregister(heap_info_.addr);
        if (err != cudaSuccess) {
            INFO(NVSHMEM_MEM, "CXL heap: cudaHostUnregister skipped: %s\n",
                 cudaGetErrorString(err));
        }
        shared_memory_close(heap_name_, &heap_info_);

        cxl_heap_slab_base_ = nullptr;
        cxl_heap_window_size_ = 0;
        heap_base_ = nullptr;
        global_heap_base_ = nullptr;

        return 0;
    }

    /* All PEs on the node map the same slab at allocation time; the peer's
     * window base in *my* mapping is what device puts and mapped RMA use. */
    int map_heap_range_by_pe(int pe_id, int transport_idx, char *buf __attribute__((unused)),
                             size_t size __attribute__((unused))) override {
        nvshmemi_state_t *state = get_state();
        if (empty_heap_handle_cache()) {
            peer_heap_base_p2p_[state->mype] = heap_base_;
            peer_heap_base_p2p_[pe_id] =
                (char *)global_heap_base_ + (pe_id % state->npes_node) * heap_size_;
        }
        return 0;
    }

    int exchange_heap_memory_handle(nvshmem_mem_handle_t *local_handles
                                    __attribute__((unused))) override {
        return 0; /* slab is already mapped for all node PEs */
    }

    int register_heap_memory_handle(nvshmem_mem_handle_t *local_handles, int transport_idx,
                                    void *buf, size_t size,
                                    nvshmem_transport_t current) override {
        /* The CXL transport needs no registration either; give it the address
         * and length so its handle encoding stays self-consistent. */
        if (local_handles && current != nullptr) {
            nvshmem_mem_handle_t *h = local_handles + transport_idx;
            MEM_HANDLE_DATA(h)[0] = (uint64_t)(uintptr_t)buf;
            MEM_HANDLE_DATA(h)[1] = size;
        }
        return 0;
    }

    int export_memory(nvshmem_mem_handle_t *mem_handle __attribute__((unused)),
                      void *buf __attribute__((unused)),
                      size_t length __attribute__((unused))) override {
        return 0; /* entire slab is mapped for all PEs at allocation time */
    }

    int import_memory(nvshmem_mem_handle_t *mem_handle __attribute__((unused)),
                      void **buf __attribute__((unused)),
                      size_t size __attribute__((unused))) override {
        return 0; /* entire slab is mapped for all PEs at allocation time */
    }

    int release_memory(void *buf __attribute__((unused)),
                       size_t size __attribute__((unused))) override {
        return 0; /* munmap happens once, at heap cleanup time */
    }

   private:
    char heap_name_[NAME_MAX] = {0};
    nvshmemi_shared_memory_info_t heap_info_ = {};
    static std::vector<nvshmemi_shared_memory_info_t> infos_;
};

std::vector<nvshmemi_shared_memory_info_t> nvshmemi_symmetric_heap_cxl_type3::infos_;

/*
 * Factory function to create CXL Type 3 heap
 */
extern "C" nvshmemi_symmetric_heap *nvshmemi_create_cxl_type3_heap(nvshmemi_state_t *state) {
    return new nvshmemi_symmetric_heap_cxl_type3(state);
}
