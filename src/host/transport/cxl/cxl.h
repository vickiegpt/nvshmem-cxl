/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#ifndef _CXL_H
#define _CXL_H

#include <stdint.h>
#include <cuda.h>
#include <map>
#include <vector>
#include "internal/host_transport/nvshmemi_transport_defines.h"

/* Forward declarations - actual definitions are in transport.h
 * Include transport.h in .cpp files that need the full definitions */
struct nvshmem_transport;
struct nvshmem_transport_pe_info;
struct rma_verb;
struct rma_memdesc;
struct rma_bytesdesc;
struct amo_verb;
struct amo_memdesc;
struct amo_bytesdesc;

/* Typedef for transport pointer (matches transport.h) */
typedef struct nvshmem_transport *nvshmem_transport_t;

/*
 * NVIDIA RM (Resource Manager) ioctl definitions for CXL P2P DMA
 * These are used to communicate with the NVIDIA driver for CXL operations
 */
#define NV_IOCTL_MAGIC 'F'
#define NV_ESC_RM_CONTROL _IOWR(NV_IOCTL_MAGIC, 0x2a, NVOS54_PARAMETERS)
#define NV_ESC_RM_ALLOC   _IOWR(NV_IOCTL_MAGIC, 0x2b, NVOS21_PARAMETERS)
#define NV_ESC_RM_FREE    _IOWR(NV_IOCTL_MAGIC, 0x29, NVOS00_PARAMETERS)

/* RM Control commands for CXL operations */
#define NV0000_CTRL_CMD_GPU_GET_PROBED_IDS    0x00000214
#define NV0000_CTRL_CMD_GPU_ATTACH_IDS        0x00000215
#define NV2080_CTRL_CMD_BUS_GET_CXL_INFO          0x20801833
#define NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST   0x20801834
#define NV2080_CTRL_CMD_BUS_REGISTER_CXL_BUFFER   0x20801835
#define NV2080_CTRL_CMD_BUS_UNREGISTER_CXL_BUFFER 0x20801836

/* RM allocation classes */
#define NV01_ROOT        0x00000000
#define NV01_DEVICE_0    0x00000080
#define NV20_SUBDEVICE_0 0x00002080

/* CXL P2P DMA direction flags */
#define CXL_P2P_DMA_FLAG_GPU_TO_CXL  0x0
#define CXL_P2P_DMA_FLAG_CXL_TO_GPU  0x1

/* P2P DMA threshold - use P2P DMA for transfers > 256KB
 * Using different name from env var CXL_P2P_DMA_THRESHOLD to avoid conflicts */
#define CXL_P2P_DMA_THRESHOLD_BYTES (256 * 1024)

/* CXL device types */
typedef enum {
    NVSHMEMI_CXL_TYPE_NONE = 0,
    NVSHMEMI_CXL_TYPE_1 = 1,    /* CXL.io only (PCIe-like) */
    NVSHMEMI_CXL_TYPE_2 = 2,    /* CXL.cache + CXL.mem (GPU with HDM) */
    NVSHMEMI_CXL_TYPE_3 = 3,    /* CXL.mem only (memory expander) */
    NVSHMEMI_CXL_TYPE_2_3 = 4   /* Type 2 + Type 3 combined */
} nvshmemi_cxl_device_type_t;

/* RM ioctl parameter structures */
typedef struct {
    uint32_t hRoot;
    uint32_t hObjectParent;
    uint32_t hObjectNew;
    uint32_t hClass;
    uint64_t pAllocParms;
    uint32_t paramsSize;
    uint32_t status;
} NVOS21_PARAMETERS;

typedef struct {
    uint32_t hClient;
    uint32_t hObject;
    uint32_t cmd;
    uint32_t flags;
    uint64_t params;
    uint32_t paramsSize;
    uint32_t status;
} NVOS54_PARAMETERS;

typedef struct {
    uint32_t hRoot;
    uint32_t hObjectParent;
    uint32_t hObjectOld;
    uint32_t status;
} NVOS00_PARAMETERS;

/* GPU probed IDs parameters */
typedef struct {
    uint32_t gpuIds[32];
    uint32_t excludedGpuIds[32];
} NV0000_CTRL_GPU_GET_PROBED_IDS_PARAMS;

/* GPU attach parameters */
typedef struct {
    uint32_t gpuIds[32];
    uint32_t failedGpuId;
} NV0000_CTRL_GPU_ATTACH_IDS_PARAMS;

#define NV0000_CTRL_GPU_ATTACH_ALL_PROBED_IDS 0xFFFFFFFF

/* Device allocation parameters */
typedef struct {
    uint32_t deviceId;
    uint32_t hClientShare;
    uint32_t hTargetClient;
    uint32_t hTargetDevice;
    uint32_t flags;
    uint64_t vaSpaceSize;
    uint64_t vaStartInternal;
    uint64_t vaLimitInternal;
    uint32_t vaMode;
} NV0080_ALLOC_PARAMETERS;

/* Subdevice allocation parameters */
typedef struct {
    uint32_t subDeviceId;
} NV2080_ALLOC_PARAMETERS;

/* CXL info query parameters */
typedef struct {
    uint8_t  bIsLinkUp;           /* CXL link status */
    uint8_t  bMemoryExpander;     /* Memory expander capability */
    uint32_t nrLinks;             /* Number of CXL links */
    uint32_t maxNrLinks;          /* Maximum links supported */
    uint32_t linkMask;            /* Mask of active links */
    uint32_t perLinkBwMBps;       /* Bandwidth per link (MB/s) */
    uint32_t cxlVersion;          /* CXL version (1.x, 2.x) */
    uint32_t remoteType;          /* Type of remote device (Type 2/3) */
} NV2080_CTRL_CMD_BUS_GET_CXL_INFO_PARAMS;

/* CXL buffer registration parameters */
typedef struct {
    uint64_t baseAddress;      /* CPU virtual address of buffer */
    uint64_t size;             /* Size of buffer */
    uint32_t cxlVersion;       /* CXL version to use */
    uint64_t bufferHandle;     /* Returned handle from driver */
} NV2080_CTRL_CMD_BUS_REGISTER_CXL_BUFFER_PARAMS;

/* CXL buffer unregistration parameters */
typedef struct {
    uint64_t bufferHandle;     /* Handle to unregister */
} NV2080_CTRL_CMD_BUS_UNREGISTER_CXL_BUFFER_PARAMS;

/* CXL P2P DMA request parameters */
typedef struct {
    uint64_t cxlBufferHandle;    /* Handle to registered CXL buffer */
    uint64_t gpuOffset;          /* Offset in GPU memory */
    uint64_t cxlOffset;          /* Offset in CXL memory */
    uint64_t size;               /* Size of transfer in bytes */
    uint32_t flags;              /* Direction flags (GPU_TO_CXL or CXL_TO_GPU) */
    uint32_t transferId;         /* Returned transfer ID for tracking */
} NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST_PARAMS;

/* CXL buffer tracking structure */
typedef struct {
    void *cpuPtr;              /* CPU virtual address */
    uint64_t size;             /* Buffer size */
    uint64_t driverHandle;     /* Handle from driver (for P2P DMA) */
    bool isRegistered;         /* Registration status */
} cxl_buffer_t;

/* CXL device info structure */
typedef struct {
    nvshmemi_cxl_device_type_t cxl_type;  /* CXL device type */
    uint64_t hdm_base;               /* Host-managed Device Memory base */
    size_t hdm_size;                 /* HDM size */
    uint32_t cxl_dvsec_vendor_id;    /* CXL DVSEC capability */
    pcie_id_t pcie_id;               /* PCIe BDF */
    bool cache_coherent;             /* CXL.cache support */
    bool back_invalidate;            /* Back-invalidate snoop support */
    int dax_fd;                      /* DAX device file descriptor */
    char dax_path[256];              /* DAX device path */
} cxl_device_info_t;

/* CXL RM context structure */
typedef struct {
    int ctlFd;                  /* File descriptor for /dev/nvidiactl */
    int devFd;                  /* File descriptor for /dev/nvidia0 */
    uint32_t hClient;           /* RM client handle */
    uint32_t hDevice;           /* GPU device handle */
    uint32_t hSubdevice;        /* GPU subdevice handle */
    uint32_t cxlVersion;        /* CXL version (e.g., 2 for CXL 2.0) */
    bool initialized;           /* Initialization flag */
    bool cxl_link_up;           /* CXL link status */
    bool p2p_dma_available;     /* P2P DMA availability */
} cxl_rm_context_t;

/* CXL transport state structure */
typedef struct {
    int ndev;                        /* Number of CXL devices */
    int n_type2_dev;                 /* Number of Type 2 devices (GPUs) */
    int n_type3_dev;                 /* Number of Type 3 devices (memory) */
    cxl_device_info_t *devices;      /* Device info array */
    CUdevice cudevice;               /* Current CUDA device */
    int device_id;                   /* Device ID */
    uint64_t hostHash;
    pcie_id_t *pcie_ids;             /* PCIe IDs for all devices */
    char pcie_bdf[NVSHMEM_PCIE_BDF_BUFFER_LEN];

    /* Type 3 memory pool management */
    void *type3_pool_base;           /* Base address of Type 3 memory pool */
    size_t type3_pool_size;          /* Total Type 3 memory size */
    int *type3_dev_indices;          /* Indices of Type 3 devices */

    /* GPU as Type 2 device info */
    void *gpu_hdm_base;              /* GPU HDM (CXL-exposed VRAM) */
    size_t gpu_hdm_size;
    bool gpu_cxl_capable;            /* GPU supports CXL.mem */

    /* RM context for P2P DMA */
    cxl_rm_context_t rm_ctx;

    /* Buffer registry */
    std::map<uint64_t, cxl_buffer_t> *buffers;
    uint64_t next_buffer_id;

    /* CXL info from driver */
    NV2080_CTRL_CMD_BUS_GET_CXL_INFO_PARAMS cxl_info;
} transport_cxl_state_t;

/* Transport initialization */
int nvshmemt_cxl_init(nvshmem_transport_t *transport);

/* Transport operations */
int nvshmemt_cxl_can_reach_peer(int *access, struct nvshmem_transport_pe_info *peer_info,
                                 nvshmem_transport_t transport);
int nvshmemt_cxl_connect_endpoints(struct nvshmem_transport *tcurr, int *selected_dev_ids,
                                    int num_selected_devs, int *out_qp_indices, int num_qps);
int nvshmemt_cxl_get_mem_handle(nvshmem_mem_handle_t *mem_handle, void *buf, size_t size,
                                 struct nvshmem_transport *transport, bool local_only);
int nvshmemt_cxl_release_mem_handle(nvshmem_mem_handle_t *mem_handle,
                                     struct nvshmem_transport *transport);
int nvshmemt_cxl_finalize(nvshmem_transport_t transport);
int nvshmemt_cxl_show_info(struct nvshmem_transport *transport, int style);

/* RMA operations - use struct types as forward-declared above */
int nvshmemt_cxl_rma(struct nvshmem_transport *tcurr, int pe, struct rma_verb verb,
                     struct rma_memdesc *remote, struct rma_memdesc *local,
                     struct rma_bytesdesc bytesdesc, int qp_index);
int nvshmemt_cxl_amo(struct nvshmem_transport *tcurr, int pe, void *curetptr,
                     struct amo_verb verb, struct amo_memdesc *target,
                     struct amo_bytesdesc bytesdesc, int qp_index);
int nvshmemt_cxl_fence(struct nvshmem_transport *tcurr, int pe, int qp_index, int is_multi);
int nvshmemt_cxl_quiet(struct nvshmem_transport *tcurr, int pe, int qp_index);

/* P2P DMA operations (from cxl_pytorch_expander) */
int cxl_rm_control(cxl_rm_context_t *ctx, uint32_t hObject, uint32_t cmd,
                   void *params, uint32_t paramsSize);
int cxl_rm_alloc(cxl_rm_context_t *ctx, uint32_t hParent, uint32_t hObject,
                 uint32_t hClass, void *allocParams, uint32_t paramsSize);
int cxl_rm_free(cxl_rm_context_t *ctx, uint32_t hParent, uint32_t hObject);

/* Buffer management */
uint64_t cxl_alloc_buffer(transport_cxl_state_t *state, size_t size);
int cxl_free_buffer(transport_cxl_state_t *state, uint64_t buffer_id);
void *cxl_get_buffer_ptr(transport_cxl_state_t *state, uint64_t buffer_id);
size_t cxl_get_buffer_size(transport_cxl_state_t *state, uint64_t buffer_id);

/* P2P DMA transfers */
int cxl_gpu_to_cxl(transport_cxl_state_t *state, uint64_t buffer_id,
                   uint64_t gpu_ptr, uint64_t cxl_offset, size_t size);
int cxl_cxl_to_gpu(transport_cxl_state_t *state, uint64_t buffer_id,
                   uint64_t gpu_ptr, uint64_t cxl_offset, size_t size);

/* Type 3 memory mapping */
int cxl_map_type3_to_gpu(transport_cxl_state_t *state, void **gpu_va,
                         void *type3_addr, size_t size, int type3_dev_idx);
int cxl_unmap_type3_from_gpu(transport_cxl_state_t *state, void *gpu_va, size_t size);

#endif /* _CXL_H */
