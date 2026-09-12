/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 *
 * CXL Transport Implementation for NVSHMEM
 * Enables GPU (Type 2) to CXL memory (Type 3) P2P DMA transfers
 * Based on cxl_pytorch_expander P2P DMA mechanism
 */

#include "cxl.h"
#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>
#include "internal/host_transport/cudawrap.h"
#include "non_abi/nvshmemx_error.h"
#include "internal/host/debug.h"
#include "internal/host/nvshmem_internal.h"
#include "internal/host/nvshmemi_symmetric_heap.hpp"
#include "internal/host/nvshmemi_mem_transport.hpp"
#include "internal/host/nvshmemi_types.h"
#include "internal/host/util.h"
#include "internal/host_transport/transport.h"

/* Helper macro to access mem_handle as uint64_t array
 * nvshmem_mem_handle_t has 'reserved' char array, we store uint64_t values in it */
#define MEM_HANDLE_DATA(h) ((uint64_t*)((h)->reserved))

/* Static helper to get wall time in nanoseconds */
static inline uint64_t get_wall_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/*
 * RM Control - Send control command to NVIDIA driver
 * This is the core interface for CXL P2P DMA operations
 */
int cxl_rm_control(cxl_rm_context_t *ctx, uint32_t hObject, uint32_t cmd,
                   void *params, uint32_t paramsSize) {
    NVOS54_PARAMETERS ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.hClient = ctx->hClient;
    ctrl.hObject = hObject;
    ctrl.cmd = cmd;
    ctrl.flags = 0;
    ctrl.params = (uint64_t)(uintptr_t)params;
    ctrl.paramsSize = paramsSize;
    ctrl.status = 0;

    int ret = ioctl(ctx->ctlFd, NV_ESC_RM_CONTROL, &ctrl);
    if (ret < 0) {
        INFO(NVSHMEM_TRANSPORT, "RM control ioctl failed: %s\n", strerror(errno));
        return -errno;
    }
    return ctrl.status;
}

/*
 * RM Alloc - Allocate a resource in the NVIDIA driver
 */
int cxl_rm_alloc(cxl_rm_context_t *ctx, uint32_t hParent, uint32_t hObject,
                 uint32_t hClass, void *allocParams, uint32_t paramsSize) {
    NVOS21_PARAMETERS alloc;
    memset(&alloc, 0, sizeof(alloc));
    alloc.hRoot = ctx->hClient;
    alloc.hObjectParent = hParent;
    alloc.hObjectNew = hObject;
    alloc.hClass = hClass;
    alloc.pAllocParms = (uint64_t)(uintptr_t)allocParams;
    alloc.paramsSize = paramsSize;
    alloc.status = 0;

    int ret = ioctl(ctx->ctlFd, NV_ESC_RM_ALLOC, &alloc);
    if (ret < 0) {
        INFO(NVSHMEM_TRANSPORT, "RM alloc ioctl failed: %s\n", strerror(errno));
        return -errno;
    }
    return alloc.status;
}

/*
 * RM Free - Free a resource in the NVIDIA driver
 */
int cxl_rm_free(cxl_rm_context_t *ctx, uint32_t hParent, uint32_t hObject) {
    NVOS00_PARAMETERS free_params;
    memset(&free_params, 0, sizeof(free_params));
    free_params.hRoot = ctx->hClient;
    free_params.hObjectParent = hParent;
    free_params.hObjectOld = hObject;
    free_params.status = 0;

    int ret = ioctl(ctx->ctlFd, NV_ESC_RM_FREE, &free_params);
    if (ret < 0) {
        INFO(NVSHMEM_TRANSPORT, "RM free ioctl failed: %s\n", strerror(errno));
        return -errno;
    }
    return free_params.status;
}

/*
 * Initialize RM context for CXL operations
 */
static int cxl_init_rm_context(cxl_rm_context_t *ctx, int device_id) {
    int status = 0;
    char dev_path[64];

    /* Open NVIDIA control device */
    ctx->ctlFd = open("/dev/nvidiactl", O_RDWR);
    if (ctx->ctlFd < 0) {
        INFO(NVSHMEM_TRANSPORT, "Failed to open /dev/nvidiactl: %s\n", strerror(errno));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    /* Open specific GPU device */
    snprintf(dev_path, sizeof(dev_path), "/dev/nvidia%d", device_id);
    ctx->devFd = open(dev_path, O_RDWR);
    if (ctx->devFd < 0) {
        INFO(NVSHMEM_TRANSPORT, "Failed to open %s: %s\n", dev_path, strerror(errno));
        close(ctx->ctlFd);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    /* Allocate RM client (root object) */
    ctx->hClient = 0xC1010001;  /* Unique client handle for CXL transport */
    status = cxl_rm_alloc(ctx, 0, ctx->hClient, NV01_ROOT, NULL, 0);
    if (status != 0) {
        INFO(NVSHMEM_TRANSPORT, "Failed to allocate RM client: status=%d\n", status);
        close(ctx->devFd);
        close(ctx->ctlFd);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    /* Get probed GPU IDs */
    NV0000_CTRL_GPU_GET_PROBED_IDS_PARAMS probedParams;
    memset(&probedParams, 0, sizeof(probedParams));
    status = cxl_rm_control(ctx, ctx->hClient, NV0000_CTRL_CMD_GPU_GET_PROBED_IDS,
                            &probedParams, sizeof(probedParams));
    if (status != 0) {
        INFO(NVSHMEM_TRANSPORT, "Failed to get probed GPU IDs: status=%d\n", status);
        goto cleanup;
    }

    /* Attach all probed GPUs */
    NV0000_CTRL_GPU_ATTACH_IDS_PARAMS attachParams;
    memset(&attachParams, 0, sizeof(attachParams));
    attachParams.gpuIds[0] = NV0000_CTRL_GPU_ATTACH_ALL_PROBED_IDS;
    status = cxl_rm_control(ctx, ctx->hClient, NV0000_CTRL_CMD_GPU_ATTACH_IDS,
                            &attachParams, sizeof(attachParams));
    if (status != 0) {
        INFO(NVSHMEM_TRANSPORT, "Failed to attach GPUs: status=%d\n", status);
        goto cleanup;
    }

    /* Allocate GPU device object */
    NV0080_ALLOC_PARAMETERS devParams;
    memset(&devParams, 0, sizeof(devParams));
    devParams.deviceId = device_id;
    ctx->hDevice = 0xC1010002;
    status = cxl_rm_alloc(ctx, ctx->hClient, ctx->hDevice, NV01_DEVICE_0,
                          &devParams, sizeof(devParams));
    if (status != 0) {
        INFO(NVSHMEM_TRANSPORT, "Failed to allocate device object: status=%d\n", status);
        goto cleanup;
    }

    /* Allocate subdevice object */
    NV2080_ALLOC_PARAMETERS subdevParams;
    memset(&subdevParams, 0, sizeof(subdevParams));
    subdevParams.subDeviceId = 0;
    ctx->hSubdevice = 0xC1010003;
    status = cxl_rm_alloc(ctx, ctx->hDevice, ctx->hSubdevice, NV20_SUBDEVICE_0,
                          &subdevParams, sizeof(subdevParams));
    if (status != 0) {
        INFO(NVSHMEM_TRANSPORT, "Failed to allocate subdevice object: status=%d\n", status);
        goto cleanup;
    }

    ctx->initialized = true;
    return 0;

cleanup:
    if (ctx->hDevice) cxl_rm_free(ctx, ctx->hClient, ctx->hDevice);
    if (ctx->hClient) cxl_rm_free(ctx, 0, ctx->hClient);
    close(ctx->devFd);
    close(ctx->ctlFd);
    return NVSHMEMX_ERROR_INTERNAL;
}

/*
 * Query CXL capabilities from the GPU
 */
static int cxl_query_info(transport_cxl_state_t *state) {
    int status;

    memset(&state->cxl_info, 0, sizeof(state->cxl_info));
    status = cxl_rm_control(&state->rm_ctx, state->rm_ctx.hSubdevice,
                            NV2080_CTRL_CMD_BUS_GET_CXL_INFO,
                            &state->cxl_info, sizeof(state->cxl_info));

    if (status != 0) {
        INFO(NVSHMEM_TRANSPORT, "CXL info query failed or not supported: status=%d\n", status);
        state->rm_ctx.cxl_link_up = false;
        state->rm_ctx.p2p_dma_available = false;
        return status;
    }

    state->rm_ctx.cxl_link_up = state->cxl_info.bIsLinkUp;
    state->rm_ctx.cxlVersion = state->cxl_info.cxlVersion;

    INFO(NVSHMEM_TRANSPORT, "CXL Info: link_up=%d, version=%d, links=%d, bw=%d MB/s\n",
         state->cxl_info.bIsLinkUp, state->cxl_info.cxlVersion,
         state->cxl_info.nrLinks, state->cxl_info.perLinkBwMBps);

    return 0;
}

/*
 * Discover CXL Type 3 memory devices via PCIe P2P
 * CXL memory is accessed through PCIe BAR, not DAX devices
 * We detect CXL capability via NVIDIA driver's CXL info query
 */
static int discover_cxl_type3_devices(transport_cxl_state_t *state) {
    /* CXL Type 3 memory is accessed via PCIe P2P through the GPU
     * The NVIDIA driver handles the CXL link management
     * We just need to verify CXL link is up via the RM query
     */

    if (!state->rm_ctx.cxl_link_up) {
        INFO(NVSHMEM_TRANSPORT, "CXL link not up, PCIe P2P to CXL memory not available\n");
        return 0;
    }

    /* Allocate device info for the CXL memory expander */
    state->devices = (cxl_device_info_t *)calloc(1, sizeof(cxl_device_info_t));
    if (!state->devices) {
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }

    state->type3_dev_indices = (int *)calloc(1, sizeof(int));
    if (!state->type3_dev_indices) {
        free(state->devices);
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }

    /* Set up CXL Type 3 device info based on driver query */
    state->devices[0].cxl_type = NVSHMEMI_CXL_TYPE_3;
    state->devices[0].hdm_size = 0;  /* Will be determined by buffer allocation */
    state->devices[0].cache_coherent = (state->cxl_info.cxlVersion >= 2);
    state->devices[0].back_invalidate = state->cxl_info.bMemoryExpander;
    if (nvshmemi_options.CXL_FORCE) {
        /* Forced NUMA-tier mode: the tier is ordinary cache-coherent host RAM
         * on the CXL node; CPU and GPU atomics against it are coherent. */
        state->devices[0].cache_coherent = true;
        state->devices[0].back_invalidate = true;
        snprintf(state->devices[0].dax_path, sizeof(state->devices[0].dax_path),
                 "numa_node%d", nvshmemi_cxl_heap_numa_node());
    }
    state->devices[0].dax_fd = -1;  /* Not using DAX */

    state->type3_dev_indices[0] = 0;
    state->n_type3_dev = 1;
    state->ndev = 1;

    INFO(NVSHMEM_TRANSPORT, "CXL Type 3 memory available via PCIe P2P: version=%d, bw=%d MB/s, links=%d\n",
         state->cxl_info.cxlVersion, state->cxl_info.perLinkBwMBps, state->cxl_info.nrLinks);

    return 0;
}

/*
 * Allocate a CXL buffer with P2P DMA capability
 */
uint64_t cxl_alloc_buffer(transport_cxl_state_t *state, size_t size) {
    void *ptr = NULL;
    int status;

    /* Try to allocate with huge pages first */
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);

    if (ptr == MAP_FAILED) {
        /* Fallback to regular pages */
        ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            INFO(NVSHMEM_TRANSPORT, "Failed to allocate CXL buffer: %s\n", strerror(errno));
            return 0;
        }
    }

    /* Lock pages in memory */
    if (mlock(ptr, size) != 0) {
        INFO(NVSHMEM_TRANSPORT, "Warning: mlock failed: %s\n", strerror(errno));
    }

    /* Touch all pages to ensure physical allocation */
    memset(ptr, 0, size);

    /* Register with NVIDIA driver for P2P DMA */
    NV2080_CTRL_CMD_BUS_REGISTER_CXL_BUFFER_PARAMS regParams;
    memset(&regParams, 0, sizeof(regParams));
    regParams.baseAddress = (uint64_t)(uintptr_t)ptr;
    regParams.size = size;
    regParams.cxlVersion = state->rm_ctx.cxlVersion;

    status = cxl_rm_control(&state->rm_ctx, state->rm_ctx.hSubdevice,
                            NV2080_CTRL_CMD_BUS_REGISTER_CXL_BUFFER,
                            &regParams, sizeof(regParams));

    cxl_buffer_t buf;
    buf.cpuPtr = ptr;
    buf.size = size;
    buf.isRegistered = (status == 0);
    buf.driverHandle = (status == 0) ? regParams.bufferHandle : 0;

    uint64_t buffer_id = state->next_buffer_id++;
    (*state->buffers)[buffer_id] = buf;

    if (status != 0) {
        INFO(NVSHMEM_TRANSPORT, "Warning: CXL buffer registration failed, "
             "falling back to CUDA mapped memory\n");
    }

    return buffer_id;
}

/*
 * Free a CXL buffer
 */
int cxl_free_buffer(transport_cxl_state_t *state, uint64_t buffer_id) {
    auto it = state->buffers->find(buffer_id);
    if (it == state->buffers->end()) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    cxl_buffer_t &buf = it->second;

    /* Unregister from driver if registered */
    if (buf.isRegistered && buf.driverHandle != 0) {
        NV2080_CTRL_CMD_BUS_UNREGISTER_CXL_BUFFER_PARAMS unregParams;
        unregParams.bufferHandle = buf.driverHandle;
        cxl_rm_control(&state->rm_ctx, state->rm_ctx.hSubdevice,
                       NV2080_CTRL_CMD_BUS_UNREGISTER_CXL_BUFFER,
                       &unregParams, sizeof(unregParams));
    }

    /* Unlock and unmap memory */
    munlock(buf.cpuPtr, buf.size);
    munmap(buf.cpuPtr, buf.size);

    state->buffers->erase(it);
    return 0;
}

/*
 * Get buffer CPU pointer
 */
void *cxl_get_buffer_ptr(transport_cxl_state_t *state, uint64_t buffer_id) {
    auto it = state->buffers->find(buffer_id);
    if (it == state->buffers->end()) {
        return NULL;
    }
    return it->second.cpuPtr;
}

/*
 * Get buffer size
 */
size_t cxl_get_buffer_size(transport_cxl_state_t *state, uint64_t buffer_id) {
    auto it = state->buffers->find(buffer_id);
    if (it == state->buffers->end()) {
        return 0;
    }
    return it->second.size;
}

/*
 * P2P DMA: GPU to CXL transfer
 *
 * Data transfer is done via CUDA memcpy through the registered host memory.
 * The CXL memory is registered with cudaHostRegister, allowing the GPU to
 * access it via PCIe P2P. This is the actual data path for GPU<->CXL transfers.
 *
 * Note: RM control commands (NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST) are for
 * DMA engine configuration, not for actual data movement.
 */
int cxl_gpu_to_cxl(transport_cxl_state_t *state, uint64_t buffer_id,
                   uint64_t gpu_ptr, uint64_t cxl_offset, size_t size) {
    auto it = state->buffers->find(buffer_id);
    if (it == state->buffers->end()) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    cxl_buffer_t &buf = it->second;

    /* Transfer data using CUDA memcpy through registered host memory
     * The CXL memory was registered with cudaHostRegister, enabling
     * GPU DMA access via PCIe P2P */
    void *dst = (char *)buf.cpuPtr + cxl_offset;
    cudaError_t err = cudaMemcpy(dst, (void *)gpu_ptr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        INFO(NVSHMEM_TRANSPORT, "CUDA memcpy D2H failed: %s\n", cudaGetErrorString(err));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    /* Ensure transfer completes before returning */
    cudaDeviceSynchronize();

    return 0;
}

/*
 * P2P DMA: CXL to GPU transfer
 */
int cxl_cxl_to_gpu(transport_cxl_state_t *state, uint64_t buffer_id,
                   uint64_t gpu_ptr, uint64_t cxl_offset, size_t size) {
    auto it = state->buffers->find(buffer_id);
    if (it == state->buffers->end()) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    cxl_buffer_t &buf = it->second;

    /* Check if P2P DMA is available and beneficial */
    if (buf.isRegistered && buf.driverHandle != 0 && size >= CXL_P2P_DMA_THRESHOLD_BYTES) {
        /* Use P2P DMA */
        NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST_PARAMS dmaParams;
        memset(&dmaParams, 0, sizeof(dmaParams));
        dmaParams.cxlBufferHandle = buf.driverHandle;
        dmaParams.gpuOffset = gpu_ptr;
        dmaParams.cxlOffset = cxl_offset;
        dmaParams.size = size;
        dmaParams.flags = CXL_P2P_DMA_FLAG_CXL_TO_GPU;

        int status = cxl_rm_control(&state->rm_ctx, state->rm_ctx.hSubdevice,
                                    NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST,
                                    &dmaParams, sizeof(dmaParams));
        if (status == 0) {
            return 0;
        }
        /* Fall through to CUDA memcpy on failure */
        INFO(NVSHMEM_TRANSPORT, "P2P DMA failed, falling back to CUDA memcpy\n");
    }

    /* Fallback: Use CUDA memcpy through mapped host memory */
    void *src = (char *)buf.cpuPtr + cxl_offset;
    cudaError_t err = cudaMemcpy((void *)gpu_ptr, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        INFO(NVSHMEM_TRANSPORT, "CUDA memcpy H2D failed: %s\n", cudaGetErrorString(err));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    return 0;
}

/*
 * Map Type 3 memory into GPU address space
 */
int cxl_map_type3_to_gpu(transport_cxl_state_t *state, void **gpu_va,
                         void *type3_addr, size_t size, int type3_dev_idx) {
    cudaError_t err;

    /* Register the Type 3 memory with CUDA for GPU access */
    err = cudaHostRegister(type3_addr, size,
                           cudaHostRegisterMapped | cudaHostRegisterPortable);
    if (err != cudaSuccess) {
        INFO(NVSHMEM_TRANSPORT, "cudaHostRegister failed: %s\n", cudaGetErrorString(err));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    /* Get GPU-accessible pointer */
    err = cudaHostGetDevicePointer(gpu_va, type3_addr, 0);
    if (err != cudaSuccess) {
        cudaHostUnregister(type3_addr);
        INFO(NVSHMEM_TRANSPORT, "cudaHostGetDevicePointer failed: %s\n", cudaGetErrorString(err));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    return 0;
}

/*
 * Unmap Type 3 memory from GPU address space
 */
int cxl_unmap_type3_from_gpu(transport_cxl_state_t *state, void *gpu_va, size_t size) {
    /* Note: We need to track the original host pointer to unregister */
    cudaError_t err = cudaHostUnregister(gpu_va);
    if (err != cudaSuccess) {
        INFO(NVSHMEM_TRANSPORT, "cudaHostUnregister failed: %s\n", cudaGetErrorString(err));
        return NVSHMEMX_ERROR_INTERNAL;
    }
    return 0;
}

/*
 * Check if peer PE is reachable via CXL transport
 */
int nvshmemt_cxl_can_reach_peer(int *access, struct nvshmem_transport_pe_info *peer_info,
                                 nvshmem_transport_t transport) {
    transport_cxl_state_t *cxl_state = (transport_cxl_state_t *)transport->state;

    /* If CXL link is not up, cannot reach any peer */
    if (!cxl_state->rm_ctx.cxl_link_up) {
        *access = 0;
        return 0;
    }

    /* Check if peer is on the same host (CXL is node-local) */
    if (peer_info->hostHash != cxl_state->hostHash) {
        *access = 0;
        return 0;
    }

    /* Forced NUMA-tier mode: every same-host peer is reachable through the
     * shared CXL heap slab, which both CPU and GPU address directly. */
    if (nvshmemi_options.CXL_FORCE) {
        *access = NVSHMEM_TRANSPORT_CAP_MAP |
                  NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST |
                  NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD |
                  NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS |
                  NVSHMEM_TRANSPORT_CAP_CPU_WRITE |
                  NVSHMEM_TRANSPORT_CAP_CPU_READ |
                  NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS;
        return 0;
    }

    /* Check if peer is a Type 3 device in our pool */
    for (int i = 0; i < cxl_state->n_type3_dev; i++) {
        int idx = cxl_state->type3_dev_indices[i];
        if (cxl_state->devices[idx].pcie_id.domain_id == peer_info->pcie_id.domain_id &&
            cxl_state->devices[idx].pcie_id.bus_id == peer_info->pcie_id.bus_id &&
            cxl_state->devices[idx].pcie_id.dev_id == peer_info->pcie_id.dev_id) {

            /* Type 2 (GPU) can access Type 3 memory via CXL.mem */
            *access = NVSHMEM_TRANSPORT_CAP_MAP |
                      NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST |
                      NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD |
                      NVSHMEM_TRANSPORT_CAP_CPU_WRITE |
                      NVSHMEM_TRANSPORT_CAP_CPU_READ;

            /* If back-invalidate is supported, atomics are possible */
            if (cxl_state->devices[idx].back_invalidate) {
                *access |= NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS |
                           NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS;
            }

            return 0;
        }
    }

    /* Check for same-device access (GPU to its own CXL-exposed memory) */
    if (cxl_state->gpu_cxl_capable) {
        *access = NVSHMEM_TRANSPORT_CAP_MAP |
                  NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST |
                  NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD |
                  NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS;
        return 0;
    }

    *access = 0;
    return 0;
}

/*
 * Connect endpoints (setup connections to remote PEs)
 */
int nvshmemt_cxl_connect_endpoints(struct nvshmem_transport *tcurr, int *selected_dev_ids,
                                    int num_selected_devs, int *out_qp_indices, int num_qps) {
    /* CXL is memory-mapped, no explicit endpoint connections needed */
    /* Just validate that we can access the selected devices */
    return 0;
}

/*
 * Get memory handle for a buffer
 *
 * The heap slab is mapped into every same-node PE, so a handle carries only
 * the buffer's address and length.  No buffer allocation, no copy, nothing to
 * release: the receiving side resolves remote addresses through its own
 * mapping of the shared slab (see nvshmemt_cxl_rma).
 */
int nvshmemt_cxl_get_mem_handle(nvshmem_mem_handle_t *mem_handle, void *buf, size_t size,
                                 struct nvshmem_transport *transport, bool local_only) {
    memset(mem_handle, 0, sizeof(*mem_handle));
    MEM_HANDLE_DATA(mem_handle)[0] = (uint64_t)(uintptr_t)buf;
    MEM_HANDLE_DATA(mem_handle)[1] = size;

    return 0;
}

/*
 * Release memory handle
 */
int nvshmemt_cxl_release_mem_handle(nvshmem_mem_handle_t *mem_handle,
                                     struct nvshmem_transport *transport) {
    /* Handles are pure {address, length} pairs; nothing to release. */
    return 0;
}

/*
 * Translate a remote heap offset to a locally dereferenceable pointer.
 *
 * Every same-node PE maps the whole CXL heap slab; PE i's window starts at
 * slab + i * window_size (single-node PE numbering).  The rma/amo callbacks
 * only run for peers the mapped-RMA fast path could not handle, but they must
 * still land on the right physical pages.
 */
static void *cxl_local_addr_for_pe(nvshmem_transport_t tcurr, int pe, uint64_t offset) {
    void *slab = nvshmemi_cxl_heap_slab_base();
    size_t window = nvshmemi_cxl_heap_window_size();
    if (!slab || window == 0) return NULL;
    return (char *)slab + (uint64_t)pe * window + offset;
}

/*
 * RMA operation (put/get)
 */
int nvshmemt_cxl_rma(struct nvshmem_transport *tcurr, int pe, rma_verb_t verb,
                     rma_memdesc_t *remote, rma_memdesc_t *local, rma_bytesdesc_t bytesdesc,
                     int qp_index) {
    size_t size = bytesdesc.nelems * bytesdesc.elembytes;

    /* Symmetric heap traffic: resolve through my mapping of the shared slab. */
    void *remote_addr = cxl_local_addr_for_pe(tcurr, pe, remote->offset);
    if (remote_addr) {
        if (verb.desc == NVSHMEMI_OP_PUT || verb.desc == NVSHMEMI_OP_P) {
            cudaError_t err =
                cudaMemcpy(remote_addr, local->ptr, size, cudaMemcpyDefault);
            return (err == cudaSuccess) ? 0 : NVSHMEMX_ERROR_INTERNAL;
        } else if (verb.desc == NVSHMEMI_OP_GET || verb.desc == NVSHMEMI_OP_G) {
            cudaError_t err =
                cudaMemcpy(local->ptr, remote_addr, size, cudaMemcpyDefault);
            return (err == cudaSuccess) ? 0 : NVSHMEMX_ERROR_INTERNAL;
        }
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    /* Non-heap (externally registered) buffers keep the registered-buffer
     * data path. */
    transport_cxl_state_t *cxl_state = (transport_cxl_state_t *)tcurr->state;
    uint64_t buffer_id = MEM_HANDLE_DATA(remote->handle)[0];

    if (verb.desc == NVSHMEMI_OP_PUT) {
        /* PUT: local -> remote (GPU to CXL) */
        return cxl_gpu_to_cxl(cxl_state, buffer_id,
                              (uint64_t)(uintptr_t)local->ptr,
                              remote->offset, size);
    } else if (verb.desc == NVSHMEMI_OP_GET) {
        /* GET: remote -> local (CXL to GPU) */
        return cxl_cxl_to_gpu(cxl_state, buffer_id,
                              (uint64_t)(uintptr_t)local->ptr,
                              remote->offset, size);
    }

    return NVSHMEMX_ERROR_INVALID_VALUE;
}

/*
 * Atomic memory operation
 */
int nvshmemt_cxl_amo(struct nvshmem_transport *tcurr, int pe, void *curetptr, amo_verb_t verb,
                     amo_memdesc_t *target, amo_bytesdesc_t bytesdesc, int qp_index) {
    /* For CXL Type 3 with back-invalidate, atomics go through CPU */
    /* Map the target buffer and perform atomic operation */
    void *target_addr = cxl_local_addr_for_pe(tcurr, pe, target->remote_memdesc.offset);
    if (!target_addr) {
        transport_cxl_state_t *cxl_state = (transport_cxl_state_t *)tcurr->state;
        uint64_t buffer_id = MEM_HANDLE_DATA(target->remote_memdesc.handle)[0];
        void *cxl_ptr = cxl_get_buffer_ptr(cxl_state, buffer_id);
        if (!cxl_ptr) {
            return NVSHMEMX_ERROR_INVALID_VALUE;
        }
        target_addr = (char *)cxl_ptr + target->remote_memdesc.offset;
    }

    /* Perform atomic operation on CPU (CXL.cache ensures coherency) */
    switch (verb.desc) {
        case NVSHMEMI_AMO_ADD:
            if (bytesdesc.elembytes == 4) {
                __sync_fetch_and_add((uint32_t *)target_addr, (uint32_t)target->val);
            } else {
                __sync_fetch_and_add((uint64_t *)target_addr, target->val);
            }
            break;
        case NVSHMEMI_AMO_SET:
            if (bytesdesc.elembytes == 4) {
                __sync_lock_test_and_set((uint32_t *)target_addr, (uint32_t)target->val);
            } else {
                __sync_lock_test_and_set((uint64_t *)target_addr, target->val);
            }
            break;
        case NVSHMEMI_AMO_COMPARE_SWAP:
            if (bytesdesc.elembytes == 4) {
                uint32_t old = __sync_val_compare_and_swap((uint32_t *)target_addr,
                                                           (uint32_t)target->cmp,
                                                           (uint32_t)target->val);
                if (curetptr) *(uint32_t *)curetptr = old;
            } else {
                uint64_t old = __sync_val_compare_and_swap((uint64_t *)target_addr,
                                                           target->cmp, target->val);
                if (curetptr) *(uint64_t *)curetptr = old;
            }
            break;
        default:
            return NVSHMEMX_ERROR_NOT_SUPPORTED;
    }

    return 0;
}

/*
 * Fence operation
 */
int nvshmemt_cxl_fence(struct nvshmem_transport *tcurr, int pe, int qp_index, int is_multi) {
    /* CXL provides cache coherency, memory barrier ensures ordering */
    __sync_synchronize();
    return 0;
}

/*
 * Quiet operation
 */
int nvshmemt_cxl_quiet(struct nvshmem_transport *tcurr, int pe, int qp_index) {
    /* CXL provides cache coherency, memory barrier ensures completion */
    __sync_synchronize();
    return 0;
}

/*
 * Show transport info
 */
int nvshmemt_cxl_show_info(struct nvshmem_transport *transport, int style) {
    transport_cxl_state_t *cxl_state = (transport_cxl_state_t *)transport->state;

    printf("CXL Transport Info:\n");
    printf("  CXL Link Up: %s\n", cxl_state->rm_ctx.cxl_link_up ? "Yes" : "No");
    printf("  Forced NUMA-tier mode: %s\n", nvshmemi_options.CXL_FORCE ? "Yes" : "No");
    printf("  CXL Version: %d\n", cxl_state->rm_ctx.cxlVersion);
    printf("  P2P DMA Available: %s\n", cxl_state->rm_ctx.p2p_dma_available ? "Yes" : "No");
    printf("  Type 3 Devices: %d\n", cxl_state->n_type3_dev);
    printf("  Heap slab: %p (window %zu bytes, NUMA node %d)\n",
           nvshmemi_cxl_heap_slab_base(), nvshmemi_cxl_heap_window_size(),
           nvshmemi_cxl_heap_numa_node());

    for (int i = 0; i < cxl_state->n_type3_dev; i++) {
        int idx = cxl_state->type3_dev_indices[i];
        printf("    Device %d: %s, size=%zu\n", i,
               cxl_state->devices[idx].dax_path,
               cxl_state->devices[idx].hdm_size);
    }

    return 0;
}

/*
 * Finalize transport
 */
int nvshmemt_cxl_finalize(nvshmem_transport_t transport) {
    if (!transport) return 0;

    transport_cxl_state_t *cxl_state = (transport_cxl_state_t *)transport->state;
    if (cxl_state) {
        /* Free all buffers */
        if (cxl_state->buffers) {
            for (auto &pair : *cxl_state->buffers) {
                cxl_buffer_t &buf = pair.second;
                if (buf.isRegistered && buf.driverHandle != 0) {
                    NV2080_CTRL_CMD_BUS_UNREGISTER_CXL_BUFFER_PARAMS unregParams;
                    unregParams.bufferHandle = buf.driverHandle;
                    cxl_rm_control(&cxl_state->rm_ctx, cxl_state->rm_ctx.hSubdevice,
                                   NV2080_CTRL_CMD_BUS_UNREGISTER_CXL_BUFFER,
                                   &unregParams, sizeof(unregParams));
                }
                munlock(buf.cpuPtr, buf.size);
                munmap(buf.cpuPtr, buf.size);
            }
            delete cxl_state->buffers;
        }

        /* Close DAX devices */
        if (cxl_state->devices) {
            for (int i = 0; i < cxl_state->ndev; i++) {
                if (cxl_state->devices[i].dax_fd >= 0) {
                    close(cxl_state->devices[i].dax_fd);
                }
            }
            free(cxl_state->devices);
        }

        /* Free RM resources */
        if (cxl_state->rm_ctx.initialized) {
            if (cxl_state->rm_ctx.hSubdevice) {
                cxl_rm_free(&cxl_state->rm_ctx, cxl_state->rm_ctx.hDevice,
                            cxl_state->rm_ctx.hSubdevice);
            }
            if (cxl_state->rm_ctx.hDevice) {
                cxl_rm_free(&cxl_state->rm_ctx, cxl_state->rm_ctx.hClient,
                            cxl_state->rm_ctx.hDevice);
            }
            if (cxl_state->rm_ctx.hClient) {
                cxl_rm_free(&cxl_state->rm_ctx, 0, cxl_state->rm_ctx.hClient);
            }
            if (cxl_state->rm_ctx.devFd >= 0) close(cxl_state->rm_ctx.devFd);
            if (cxl_state->rm_ctx.ctlFd >= 0) close(cxl_state->rm_ctx.ctlFd);
        }

        free(cxl_state->type3_dev_indices);
        free(cxl_state->pcie_ids);
        free(cxl_state);
    }

    free(transport);
    return 0;
}

/*
 * Initialize CXL transport
 */
int nvshmemt_cxl_init(nvshmem_transport_t *t) {
    int status = 0;
    struct nvshmem_transport *transport = NULL;
    transport_cxl_state_t *cxl_state = NULL;

    transport = (struct nvshmem_transport *)malloc(sizeof(struct nvshmem_transport));
    NVSHMEMI_NULL_ERROR_JMP(transport, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "CXL transport allocation failed\n");
    memset(transport, 0, sizeof(struct nvshmem_transport));
    transport->is_successfully_initialized = false;

    cxl_state = (transport_cxl_state_t *)calloc(1, sizeof(transport_cxl_state_t));
    NVSHMEMI_NULL_ERROR_JMP(cxl_state, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "CXL state allocation failed\n");

    /* Initialize buffer map */
    cxl_state->buffers = new std::map<uint64_t, cxl_buffer_t>();
    cxl_state->next_buffer_id = 1;

    /* Get current CUDA device */
    status = CUPFN(nvshmemi_cuda_syms, cuCtxGetDevice(&cxl_state->cudevice));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuCtxGetDevice failed\n");

    cxl_state->hostHash = nvshmemu_getHostHash();

    /* Get device count and info */
    status = cudaGetDeviceCount(&cxl_state->n_type2_dev);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaGetDeviceCount failed\n");

    /* Get current device ID */
    status = cudaGetDevice(&cxl_state->device_id);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaGetDevice failed\n");

    /* Get PCIe BDF */
    cudaDeviceProp prop;
    status = cudaGetDeviceProperties(&prop, cxl_state->device_id);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaGetDeviceProperties failed\n");
    snprintf(cxl_state->pcie_bdf, NVSHMEM_PCIE_BDF_BUFFER_LEN, "%x:%x:%x.0",
             prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    /* Initialize RM context for P2P DMA */
    status = cxl_init_rm_context(&cxl_state->rm_ctx, cxl_state->device_id);
    if (status != 0) {
        INFO(NVSHMEM_TRANSPORT, "RM context init failed, CXL P2P DMA disabled\n");
        cxl_state->rm_ctx.p2p_dma_available = false;
        if (nvshmemi_options.CXL_FORCE) {
            /* Consumer drivers (GeForce) reject RM CXL queries.  In forced
             * mode the tier is host DRAM on the CXL NUMA node, which the GPU
             * reaches through the zero-copy aperture; RM is not in the path. */
            INFO(NVSHMEM_TRANSPORT,
                 "CXL_FORCE enabled: continuing without RM, NUMA-tier mode\n");
            memset(&cxl_state->rm_ctx, 0, sizeof(cxl_state->rm_ctx));
            cxl_state->rm_ctx.cxl_link_up = true;
            cxl_state->rm_ctx.p2p_dma_available = false;
        }
    } else {
        /* Query CXL capabilities */
        status = cxl_query_info(cxl_state);
        cxl_state->rm_ctx.p2p_dma_available = (status == 0 && cxl_state->rm_ctx.cxl_link_up);
        if (status != 0 && nvshmemi_options.CXL_FORCE) {
            INFO(NVSHMEM_TRANSPORT,
                 "CXL_FORCE enabled: overriding failed CXL capability query\n");
            cxl_state->rm_ctx.cxl_link_up = true;
        }
    }

    /* Discover CXL Type 3 devices */
    status = discover_cxl_type3_devices(cxl_state);
    if (status != 0) {
        INFO(NVSHMEM_TRANSPORT, "Type 3 device discovery failed\n");
    }

    /* Check if GPU has CXL capability */
    cxl_state->gpu_cxl_capable = cxl_state->rm_ctx.cxl_link_up;

    /* Set up transport operations */
    transport->host_ops.can_reach_peer = nvshmemt_cxl_can_reach_peer;
    transport->host_ops.connect_endpoints = nvshmemt_cxl_connect_endpoints;
    transport->host_ops.get_mem_handle = nvshmemt_cxl_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_cxl_release_mem_handle;
    transport->host_ops.finalize = nvshmemt_cxl_finalize;
    transport->host_ops.show_info = nvshmemt_cxl_show_info;
    transport->host_ops.rma = nvshmemt_cxl_rma;
    transport->host_ops.amo = nvshmemt_cxl_amo;
    transport->host_ops.fence = nvshmemt_cxl_fence;
    transport->host_ops.quiet = nvshmemt_cxl_quiet;

    transport->attr = 0;  /* CXL needs endpoint connections for Type 3 */
    transport->state = cxl_state;
    transport->is_successfully_initialized = true;
    transport->no_proxy = true;  /* CXL doesn't need proxy */

    INFO(NVSHMEM_TRANSPORT, "CXL transport initialized: link_up=%d, p2p_dma=%d, type3_devs=%d\n",
         cxl_state->rm_ctx.cxl_link_up, cxl_state->rm_ctx.p2p_dma_available,
         cxl_state->n_type3_dev);

    *t = transport;
    return 0;

out:
    if (cxl_state) {
        if (cxl_state->buffers) delete cxl_state->buffers;
        free(cxl_state);
    }
    if (transport) free(transport);
    return status;
}
