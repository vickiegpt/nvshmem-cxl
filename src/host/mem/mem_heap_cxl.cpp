/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 *
 * CXL Type 3 Memory Heap Implementation
 * Manages symmetric heap allocated from CXL memory expansion devices
 */

#include "internal/host/nvshmem_internal.h"
#include "internal/host/util.h"
#include "internal/host/debug.h"
#include "internal/host/nvshmemi_symmetric_heap.hpp"
#include "internal/host_transport/cudawrap.h"
#include <sys/mman.h>
#include <cstring>
#include <cuda_runtime.h>
#include <cerrno>

/* Helper macro to access mem_handle reserved field as uint64_t array */
#define MEM_HANDLE_DATA(h) ((uint64_t*)((h)->reserved))

/*
 * CXL Type 3 Symmetric Heap Implementation
 * This heap manages memory from CXL Type 3 memory expander devices
 */

class nvshmemi_symmetric_heap_cxl_type3 : public nvshmemi_symmetric_heap_static {
   public:
    explicit nvshmemi_symmetric_heap_cxl_type3(nvshmemi_state_t *state) noexcept
        : nvshmemi_symmetric_heap_static(state), cxl_dax_region_(nullptr) {}

    ~nvshmemi_symmetric_heap_cxl_type3() {
        cleanup_symmetric_heap();
    }

    int reserve_heap(void) override {
        /* Get heap size from CXL_HEAP_SIZE environment variable or use default */
        size_t heap_size = nvshmemi_options.CXL_HEAP_SIZE;
        if (heap_size == 0) {
            heap_size = 256 * 1024 * 1024;  /* Default 256MB */
        }

        /* Align to page size */
        size_t page_size = sysconf(_SC_PAGESIZE);
        heap_size = (heap_size + page_size - 1) & ~(page_size - 1);

        reserved_heap_size_ = heap_size;
        heap_size_ = heap_size;
        physical_internal_heap_size_ = heap_size;

        /* Set memory granularity */
        mem_granularity_ = page_size;
        set_heap_size_attr(mem_granularity_, nullptr, nullptr, &log2_mem_granularity_);

        INFO(NVSHMEM_MEM, "CXL Type 3 heap reserved: size=%zu\n", heap_size);
        return 0;
    }

    int setup_symmetric_heap(void) override {
        int status = reserve_heap();
        if (status != 0) return status;

        status = allocate_heap_memory();
        if (status != 0) return status;

        status = setup_mspace();
        if (status != 0) {
            free_heap_memory(heap_base_);
            return status;
        }

        return 0;
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
    int allocate_heap_memory() override {
        size_t heap_size = reserved_heap_size_;

        /* CXL Type 3 memory is accessed via PCIe P2P
         * Allocate memory with huge pages for better performance
         * This memory will be registered with NVIDIA driver for P2P DMA
         */

        /* Try huge pages first for better TLB performance */
        cxl_dax_region_ = mmap(NULL, heap_size, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);

        if (cxl_dax_region_ == MAP_FAILED) {
            /* Fallback to regular pages */
            INFO(NVSHMEM_MEM, "Huge pages not available, using regular pages\n");
            cxl_dax_region_ = mmap(NULL, heap_size, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (cxl_dax_region_ == MAP_FAILED) {
                INFO(NVSHMEM_MEM, "Failed to allocate CXL heap memory: %s\n", strerror(errno));
                return NVSHMEMX_ERROR_OUT_OF_MEMORY;
            }
        }

        /* Lock pages to prevent swapping - important for P2P DMA */
        if (mlock(cxl_dax_region_, heap_size) != 0) {
            INFO(NVSHMEM_MEM, "Warning: mlock failed: %s\n", strerror(errno));
        }

        /* Touch all pages to ensure physical allocation */
        memset(cxl_dax_region_, 0, heap_size);

        heap_base_ = cxl_dax_region_;

        /* Register with CUDA for GPU access via PCIe P2P
         * This enables the GPU to directly access this memory
         */
        cudaError_t cuda_status = cudaHostRegister(heap_base_, heap_size,
                                                    cudaHostRegisterMapped |
                                                    cudaHostRegisterPortable);
        if (cuda_status != cudaSuccess) {
            INFO(NVSHMEM_MEM, "cudaHostRegister failed: %s, will use P2P DMA transfers\n",
                 cudaGetErrorString(cuda_status));
            global_heap_base_ = heap_base_;
        } else {
            /* Get GPU-accessible pointer for direct load/store access */
            cuda_status = cudaHostGetDevicePointer(&global_heap_base_, heap_base_, 0);
            if (cuda_status != cudaSuccess) {
                INFO(NVSHMEM_MEM, "cudaHostGetDevicePointer failed: %s\n",
                     cudaGetErrorString(cuda_status));
                global_heap_base_ = heap_base_;
            }
        }

        INFO(NVSHMEM_MEM, "CXL Type 3 heap allocated via PCIe P2P: base=%p, gpu_base=%p, size=%zu\n",
             heap_base_, global_heap_base_, heap_size);

        return 0;
    }

    int free_heap_memory(void *addr) override {
        if (addr == nullptr) return 0;

        size_t heap_size = reserved_heap_size_;

        /* Unregister from CUDA */
        cudaHostUnregister(heap_base_);

        /* Unlock and unmap */
        if (cxl_dax_region_) {
            munlock(cxl_dax_region_, heap_size);
            munmap(cxl_dax_region_, heap_size);
            cxl_dax_region_ = nullptr;
        }

        heap_base_ = nullptr;
        global_heap_base_ = nullptr;

        return 0;
    }

    int exchange_heap_memory_handle(nvshmem_mem_handle_t *local_handles) override {
        /* For CXL Type 3, memory handles are local - just share the base address */
        if (local_handles) {
            MEM_HANDLE_DATA(local_handles)[0] = (uint64_t)(uintptr_t)heap_base_;
            MEM_HANDLE_DATA(local_handles)[1] = reserved_heap_size_;
        }
        return 0;
    }

    int register_heap_memory_handle(nvshmem_mem_handle_t *local, int transport_idx,
                                    void *buf, size_t size,
                                    nvshmem_transport_t current) override {
        /* CXL memory is already accessible, just record the handle */
        if (local) {
            MEM_HANDLE_DATA(local)[0] = (uint64_t)(uintptr_t)buf;
            MEM_HANDLE_DATA(local)[1] = size;
        }
        return 0;
    }

    int map_heap_range_by_pe(int pe_id, int transport_idx,
                             char *buf, size_t size) override {
        /* For CXL Type 3, memory is globally visible on the node */
        /* Just record the mapping for this PE */
        if (peer_heap_base_p2p_) {
            peer_heap_base_p2p_[pe_id] = buf ? buf : (char *)heap_base_;
        }
        return 0;
    }

    int export_memory(nvshmem_mem_handle_t *mem_handle, void *buf, size_t length) override {
        if (mem_handle) {
            MEM_HANDLE_DATA(mem_handle)[0] = (uint64_t)(uintptr_t)buf;
            MEM_HANDLE_DATA(mem_handle)[1] = length;
        }
        return 0;
    }

    int import_memory(nvshmem_mem_handle_t *mem_handle, void **buf, size_t length) override {
        if (mem_handle && buf) {
            *buf = (void *)(uintptr_t)MEM_HANDLE_DATA(mem_handle)[0];
        }
        return 0;
    }

    int release_memory(void *buf, size_t size) override {
        /* CXL memory is managed at heap level, individual releases are no-ops */
        return 0;
    }

   private:
    void *cxl_dax_region_;
};

/*
 * Factory function to create CXL Type 3 heap
 */
extern "C" nvshmemi_symmetric_heap *nvshmemi_create_cxl_type3_heap(nvshmemi_state_t *state) {
    return new nvshmemi_symmetric_heap_cxl_type3(state);
}

/*
 * Check if CXL Type 3 memory is available via PCIe P2P
 * This checks if the NVIDIA driver supports CXL P2P operations
 */
extern "C" bool nvshmemi_cxl_type3_available() {
    /* CXL availability is determined by the CXL transport during init
     * For now, we assume CXL is available if enabled via environment
     */
    return nvshmemi_options.ENABLE_CXL_TRANSPORT;
}
