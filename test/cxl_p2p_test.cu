/*
 * CXL P2P DMA Test for NVSHMEM
 * Tests GPU (Type 2) to CXL memory (Type 3) P2P transfers
 *
 * Build:
 *   nvcc -o cxl_p2p_test cxl_p2p_test.cu -I../src/include -I../src/host/transport/cxl \
 *        -L../build/src/lib -lnvshmem_host -lcuda -lcudart
 *
 * Run:
 *   export NVSHMEM_ENABLE_CXL_TRANSPORT=1
 *   ./cxl_p2p_test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <cuda_runtime.h>
#include <time.h>
#include <cstdint>
#include <cerrno>

/* NVIDIA RM ioctl definitions */
#define NV_IOCTL_MAGIC      'F'
#define NV_ESC_RM_CONTROL   _IOWR(NV_IOCTL_MAGIC, 0x2a, NVOS54_PARAMETERS)
#define NV_ESC_RM_ALLOC     _IOWR(NV_IOCTL_MAGIC, 0x2b, NVOS21_PARAMETERS)
#define NV_ESC_RM_FREE      _IOWR(NV_IOCTL_MAGIC, 0x29, NVOS00_PARAMETERS)

/* Class IDs */
#define NV01_ROOT                       0x00000000
#define NV01_DEVICE_0                   0x00000080
#define NV20_SUBDEVICE_0                0x00002080

/* Control commands */
#define NV0000_CTRL_CMD_GPU_GET_PROBED_IDS        0x00000214
#define NV0000_CTRL_CMD_GPU_ATTACH_IDS            0x00000215
#define NV2080_CTRL_CMD_BUS_GET_CXL_INFO          0x20801833
#define NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST   0x20801834
#define NV2080_CTRL_CMD_BUS_REGISTER_CXL_BUFFER   0x20801835
#define NV2080_CTRL_CMD_BUS_UNREGISTER_CXL_BUFFER 0x20801836

#define NV0000_CTRL_GPU_MAX_PROBED_GPUS           32
#define NV0000_CTRL_GPU_ATTACH_ALL_PROBED_IDS     0x0000ffff
#define NV0000_CTRL_GPU_INVALID_ID                0xffffffff

/* DMA flags */
#define CXL_P2P_DMA_FLAG_GPU_TO_CXL  0x0
#define CXL_P2P_DMA_FLAG_CXL_TO_GPU  0x1

/* Structure definitions */
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
    uint32_t hObjectNew;
    uint32_t hClass;
    uint64_t pAllocParms;
    uint32_t paramsSize;
    uint32_t status;
} NVOS21_PARAMETERS;

typedef struct {
    uint32_t hRoot;
    uint32_t hObjectParent;
    uint32_t hObjectOld;
    uint32_t status;
} NVOS00_PARAMETERS;

typedef struct {
    uint32_t deviceId;
    uint32_t hClientShare;
    uint32_t hTargetClient;
    uint32_t hTargetDevice;
    uint32_t flags;
    uint64_t vaSpaceSize __attribute__((aligned(8)));
    uint64_t vaStartInternal __attribute__((aligned(8)));
    uint64_t vaLimitInternal __attribute__((aligned(8)));
    uint32_t vaMode;
} NV0080_ALLOC_PARAMETERS;

typedef struct {
    uint32_t subDeviceId;
} NV2080_ALLOC_PARAMETERS;

typedef struct {
    uint32_t gpuIds[NV0000_CTRL_GPU_MAX_PROBED_GPUS];
    uint32_t excludedGpuIds[NV0000_CTRL_GPU_MAX_PROBED_GPUS];
} NV0000_CTRL_GPU_GET_PROBED_IDS_PARAMS;

typedef struct {
    uint32_t gpuIds[NV0000_CTRL_GPU_MAX_PROBED_GPUS];
    uint32_t failedId;
} NV0000_CTRL_GPU_ATTACH_IDS_PARAMS;

typedef struct {
    uint8_t  bIsLinkUp;
    uint8_t  bMemoryExpander;
    uint32_t nrLinks;
    uint32_t maxNrLinks;
    uint32_t linkMask;
    uint32_t perLinkBwMBps;
    uint32_t cxlVersion;
    uint32_t remoteType;
} NV2080_CTRL_CMD_BUS_GET_CXL_INFO_PARAMS;

typedef struct {
    uint64_t baseAddress;
    uint64_t size;
    uint32_t cxlVersion;
    uint64_t bufferHandle;
} NV2080_CTRL_CMD_BUS_REGISTER_CXL_BUFFER_PARAMS;

typedef struct {
    uint64_t bufferHandle;
} NV2080_CTRL_CMD_BUS_UNREGISTER_CXL_BUFFER_PARAMS;

typedef struct {
    uint64_t cxlBufferHandle;
    uint64_t gpuOffset;
    uint64_t cxlOffset;
    uint64_t size;
    uint32_t flags;
    uint32_t transferId;
} NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST_PARAMS;

/* Global context */
struct {
    int ctlFd;
    int devFd;
    uint32_t hClient;
    uint32_t hDevice;
    uint32_t hSubdevice;
    uint32_t cxlVersion;
    bool initialized;
} g_ctx = {-1, -1, 0, 0, 0, 2, false};

/* Helper functions */
static inline uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int rm_control(uint32_t hObject, uint32_t cmd, void *params, uint32_t paramsSize) {
    NVOS54_PARAMETERS ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.hClient = g_ctx.hClient;
    ctrl.hObject = hObject;
    ctrl.cmd = cmd;
    ctrl.params = (uint64_t)(uintptr_t)params;
    ctrl.paramsSize = paramsSize;

    int ret = ioctl(g_ctx.ctlFd, NV_ESC_RM_CONTROL, &ctrl);
    if (ret < 0) return -errno;
    return ctrl.status;
}

static int rm_alloc(uint32_t hParent, uint32_t hObject, uint32_t hClass,
                   void *allocParams, uint32_t paramsSize) {
    NVOS21_PARAMETERS alloc;
    memset(&alloc, 0, sizeof(alloc));

    if (hClass == NV01_ROOT) {
        alloc.hRoot = hObject;
        alloc.hObjectParent = hObject;
        alloc.hObjectNew = hObject;
    } else {
        alloc.hRoot = g_ctx.hClient;
        alloc.hObjectParent = hParent;
        alloc.hObjectNew = hObject;
    }
    alloc.hClass = hClass;
    alloc.pAllocParms = (uint64_t)(uintptr_t)allocParams;
    alloc.paramsSize = paramsSize;

    int ret = ioctl(g_ctx.ctlFd, NV_ESC_RM_ALLOC, &alloc);
    if (ret < 0) return -errno;
    return alloc.status;
}

static int rm_free(uint32_t hParent, uint32_t hObject) {
    NVOS00_PARAMETERS free_params;
    memset(&free_params, 0, sizeof(free_params));
    free_params.hRoot = g_ctx.hClient;
    free_params.hObjectParent = hParent;
    free_params.hObjectOld = hObject;

    int ret = ioctl(g_ctx.ctlFd, NV_ESC_RM_FREE, &free_params);
    if (ret < 0) return -errno;
    return free_params.status;
}

/* Initialize RM context */
static int init_rm_context() {
    int ret;

    g_ctx.ctlFd = open("/dev/nvidiactl", O_RDWR);
    if (g_ctx.ctlFd < 0) {
        printf("ERROR: Failed to open /dev/nvidiactl\n");
        return -1;
    }

    /* Allocate RM client */
    g_ctx.hClient = 0x00010001;
    ret = rm_alloc(0, g_ctx.hClient, NV01_ROOT, NULL, 0);
    if (ret != 0) {
        printf("ERROR: Failed to allocate RM client: 0x%x\n", ret);
        close(g_ctx.ctlFd);
        return -1;
    }

    /* Get probed GPUs */
    NV0000_CTRL_GPU_GET_PROBED_IDS_PARAMS probedParams;
    memset(&probedParams, 0, sizeof(probedParams));
    ret = rm_control(g_ctx.hClient, NV0000_CTRL_CMD_GPU_GET_PROBED_IDS,
                    &probedParams, sizeof(probedParams));
    if (ret != 0) {
        printf("ERROR: Failed to get probed GPUs: 0x%x\n", ret);
        return -1;
    }

    /* Attach all GPUs */
    NV0000_CTRL_GPU_ATTACH_IDS_PARAMS attachParams;
    memset(&attachParams, 0, sizeof(attachParams));
    attachParams.gpuIds[0] = NV0000_CTRL_GPU_ATTACH_ALL_PROBED_IDS;
    rm_control(g_ctx.hClient, NV0000_CTRL_CMD_GPU_ATTACH_IDS,
              &attachParams, sizeof(attachParams));

    /* Open GPU device */
    g_ctx.devFd = open("/dev/nvidia0", O_RDWR);

    /* Allocate device object */
    NV0080_ALLOC_PARAMETERS devParams;
    memset(&devParams, 0, sizeof(devParams));
    devParams.deviceId = 0;
    g_ctx.hDevice = 0x00010002;
    ret = rm_alloc(g_ctx.hClient, g_ctx.hDevice, NV01_DEVICE_0,
                  &devParams, sizeof(devParams));
    if (ret != 0) {
        printf("ERROR: Failed to allocate device: 0x%x\n", ret);
        return -1;
    }

    /* Allocate subdevice */
    NV2080_ALLOC_PARAMETERS subdevParams;
    memset(&subdevParams, 0, sizeof(subdevParams));
    g_ctx.hSubdevice = 0x00010003;
    ret = rm_alloc(g_ctx.hDevice, g_ctx.hSubdevice, NV20_SUBDEVICE_0,
                  &subdevParams, sizeof(subdevParams));
    if (ret != 0) {
        printf("ERROR: Failed to allocate subdevice: 0x%x\n", ret);
        return -1;
    }

    g_ctx.initialized = true;
    return 0;
}

/* Query CXL info */
static int query_cxl_info(NV2080_CTRL_CMD_BUS_GET_CXL_INFO_PARAMS *info) {
    memset(info, 0, sizeof(*info));
    int ret = rm_control(g_ctx.hSubdevice, NV2080_CTRL_CMD_BUS_GET_CXL_INFO,
                        info, sizeof(*info));
    return ret;
}

/* Cleanup */
static void cleanup() {
    if (g_ctx.hSubdevice) rm_free(g_ctx.hDevice, g_ctx.hSubdevice);
    if (g_ctx.hDevice) rm_free(g_ctx.hClient, g_ctx.hDevice);
    if (g_ctx.hClient) rm_free(0, g_ctx.hClient);
    if (g_ctx.devFd >= 0) close(g_ctx.devFd);
    if (g_ctx.ctlFd >= 0) close(g_ctx.ctlFd);
}

/* Test GPU to CXL P2P DMA */
static int test_p2p_dma(size_t size) {
    int ret;
    void *host_buf = NULL;
    float *gpu_buf = NULL;
    uint64_t buffer_handle = 0;

    printf("\n=== Testing P2P DMA with %zu bytes ===\n", size);

    /* Allocate host buffer (simulating CXL memory) with huge pages */
    host_buf = mmap(NULL, size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (host_buf == MAP_FAILED) {
        host_buf = mmap(NULL, size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (host_buf == MAP_FAILED) {
            printf("ERROR: Failed to allocate host buffer\n");
            return -1;
        }
        printf("Using regular pages (huge pages not available)\n");
    } else {
        printf("Using huge pages\n");
    }

    /* Lock and touch pages */
    mlock(host_buf, size);
    memset(host_buf, 0, size);

    /* Register buffer with driver */
    NV2080_CTRL_CMD_BUS_REGISTER_CXL_BUFFER_PARAMS regParams;
    memset(&regParams, 0, sizeof(regParams));
    regParams.baseAddress = (uint64_t)(uintptr_t)host_buf;
    regParams.size = size;
    regParams.cxlVersion = g_ctx.cxlVersion;

    ret = rm_control(g_ctx.hSubdevice, NV2080_CTRL_CMD_BUS_REGISTER_CXL_BUFFER,
                    &regParams, sizeof(regParams));
    if (ret != 0) {
        printf("WARNING: CXL buffer registration failed: 0x%x\n", ret);
        printf("This may indicate CXL P2P is not supported on this GPU\n");
        printf("Falling back to CUDA mapped memory test...\n");

        /* Fallback test using CUDA mapped memory */
        cudaError_t err = cudaHostRegister(host_buf, size,
                                           cudaHostRegisterMapped | cudaHostRegisterPortable);
        if (err != cudaSuccess) {
            printf("ERROR: cudaHostRegister failed: %s\n", cudaGetErrorString(err));
            munmap(host_buf, size);
            return -1;
        }

        void *gpu_mapped;
        err = cudaHostGetDevicePointer(&gpu_mapped, host_buf, 0);
        if (err != cudaSuccess) {
            printf("ERROR: cudaHostGetDevicePointer failed: %s\n", cudaGetErrorString(err));
            cudaHostUnregister(host_buf);
            munmap(host_buf, size);
            return -1;
        }

        /* Allocate GPU memory */
        err = cudaMalloc(&gpu_buf, size);
        if (err != cudaSuccess) {
            printf("ERROR: cudaMalloc failed: %s\n", cudaGetErrorString(err));
            cudaHostUnregister(host_buf);
            munmap(host_buf, size);
            return -1;
        }

        /* Initialize GPU memory */
        float *init_data = (float *)malloc(size);
        for (size_t i = 0; i < size/sizeof(float); i++) {
            init_data[i] = (float)i;
        }
        cudaMemcpy(gpu_buf, init_data, size, cudaMemcpyHostToDevice);
        free(init_data);

        /* Test GPU -> Host (via mapped memory) */
        uint64_t start = get_time_ns();
        cudaMemcpy(host_buf, gpu_buf, size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        uint64_t end = get_time_ns();

        double bw = (double)size / ((end - start) / 1e9) / (1024*1024*1024);
        printf("GPU -> Host (CUDA mapped): %zu bytes in %.3f ms (%.2f GB/s)\n",
               size, (end - start) / 1e6, bw);

        /* Verify data */
        float *verify = (float *)host_buf;
        bool correct = true;
        for (size_t i = 0; i < 10 && i < size/sizeof(float); i++) {
            if (verify[i] != (float)i) {
                printf("ERROR: Data mismatch at index %zu: expected %f, got %f\n",
                       i, (float)i, verify[i]);
                correct = false;
                break;
            }
        }
        if (correct) printf("Data verification: PASSED\n");

        /* Test Host -> GPU */
        for (size_t i = 0; i < size/sizeof(float); i++) {
            ((float *)host_buf)[i] = (float)(i * 2);
        }

        start = get_time_ns();
        cudaMemcpy(gpu_buf, host_buf, size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        end = get_time_ns();

        bw = (double)size / ((end - start) / 1e9) / (1024*1024*1024);
        printf("Host -> GPU (CUDA mapped): %zu bytes in %.3f ms (%.2f GB/s)\n",
               size, (end - start) / 1e6, bw);

        cudaFree(gpu_buf);
        cudaHostUnregister(host_buf);
        munmap(host_buf, size);
        return 0;
    }

    buffer_handle = regParams.bufferHandle;
    printf("CXL buffer registered: handle=0x%lx\n", buffer_handle);

    /* Allocate GPU memory */
    {
        cudaError_t err = cudaMalloc(&gpu_buf, size);
        if (err != cudaSuccess) {
            printf("ERROR: cudaMalloc failed: %s\n", cudaGetErrorString(err));
            goto cleanup;
        }
    }

    /* Initialize GPU memory */
    {
        float *init_data = (float *)malloc(size);
        for (size_t i = 0; i < size/sizeof(float); i++) {
            init_data[i] = (float)i;
        }
        cudaMemcpy(gpu_buf, init_data, size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        free(init_data);
    }

    /* Test GPU -> CXL P2P DMA */
    {
        NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST_PARAMS dmaParams;
        memset(&dmaParams, 0, sizeof(dmaParams));
        dmaParams.cxlBufferHandle = buffer_handle;
        dmaParams.gpuOffset = (uint64_t)(uintptr_t)gpu_buf;
        dmaParams.cxlOffset = 0;
        dmaParams.size = size;
        dmaParams.flags = CXL_P2P_DMA_FLAG_GPU_TO_CXL;

        uint64_t start = get_time_ns();
        ret = rm_control(g_ctx.hSubdevice, NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST,
                        &dmaParams, sizeof(dmaParams));
        uint64_t end = get_time_ns();

        if (ret != 0) {
            printf("WARNING: GPU->CXL P2P DMA failed: 0x%x\n", ret);
        } else {
            double bw = (double)size / ((end - start) / 1e9) / (1024*1024*1024);
            printf("GPU -> CXL (P2P DMA): %zu bytes in %.3f ms (%.2f GB/s)\n",
                   size, (end - start) / 1e6, bw);

            /* Verify data */
            float *verify = (float *)host_buf;
            bool correct = true;
            for (size_t i = 0; i < 10 && i < size/sizeof(float); i++) {
                if (verify[i] != (float)i) {
                    printf("ERROR: Data mismatch at index %zu\n", i);
                    correct = false;
                    break;
                }
            }
            if (correct) printf("Data verification: PASSED\n");
        }
    }

    /* Test CXL -> GPU P2P DMA */
    {
        /* Modify host buffer */
        float *host_data = (float *)host_buf;
        for (size_t i = 0; i < size/sizeof(float); i++) {
            host_data[i] = (float)(i * 2);
        }

        NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST_PARAMS dmaParams;
        memset(&dmaParams, 0, sizeof(dmaParams));
        dmaParams.cxlBufferHandle = buffer_handle;
        dmaParams.gpuOffset = (uint64_t)(uintptr_t)gpu_buf;
        dmaParams.cxlOffset = 0;
        dmaParams.size = size;
        dmaParams.flags = CXL_P2P_DMA_FLAG_CXL_TO_GPU;

        uint64_t start = get_time_ns();
        ret = rm_control(g_ctx.hSubdevice, NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST,
                        &dmaParams, sizeof(dmaParams));
        uint64_t end = get_time_ns();

        if (ret != 0) {
            printf("WARNING: CXL->GPU P2P DMA failed: 0x%x\n", ret);
        } else {
            double bw = (double)size / ((end - start) / 1e9) / (1024*1024*1024);
            printf("CXL -> GPU (P2P DMA): %zu bytes in %.3f ms (%.2f GB/s)\n",
                   size, (end - start) / 1e6, bw);

            /* Verify by copying back to host */
            float *verify_buf = (float *)malloc(size);
            cudaMemcpy(verify_buf, gpu_buf, size, cudaMemcpyDeviceToHost);
            bool correct = true;
            for (size_t i = 0; i < 10 && i < size/sizeof(float); i++) {
                if (verify_buf[i] != (float)(i * 2)) {
                    printf("ERROR: Data mismatch at index %zu\n", i);
                    correct = false;
                    break;
                }
            }
            if (correct) printf("Data verification: PASSED\n");
            free(verify_buf);
        }
    }

cleanup:
    /* Unregister buffer */
    if (buffer_handle) {
        NV2080_CTRL_CMD_BUS_UNREGISTER_CXL_BUFFER_PARAMS unregParams;
        unregParams.bufferHandle = buffer_handle;
        rm_control(g_ctx.hSubdevice, NV2080_CTRL_CMD_BUS_UNREGISTER_CXL_BUFFER,
                  &unregParams, sizeof(unregParams));
    }

    if (gpu_buf) cudaFree(gpu_buf);
    munlock(host_buf, size);
    munmap(host_buf, size);

    return 0;
}

int main(int argc, char **argv) {
    printf("=== CXL P2P DMA Test for NVSHMEM ===\n\n");

    /* Initialize CUDA */
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        printf("ERROR: No CUDA devices found\n");
        return 1;
    }
    printf("Found %d CUDA device(s)\n", deviceCount);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s\n", prop.name);
    printf("PCIe: %04x:%02x:%02x.0\n", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    /* Initialize RM context */
    if (init_rm_context() != 0) {
        printf("ERROR: Failed to initialize RM context\n");
        return 1;
    }
    printf("RM context initialized successfully\n");

    /* Query CXL info */
    NV2080_CTRL_CMD_BUS_GET_CXL_INFO_PARAMS cxlInfo;
    int ret = query_cxl_info(&cxlInfo);
    if (ret != 0) {
        printf("WARNING: CXL info query failed: 0x%x\n", ret);
        printf("CXL may not be supported on this GPU or driver version\n");
    } else {
        printf("\nCXL Info:\n");
        printf("  Link Up: %s\n", cxlInfo.bIsLinkUp ? "Yes" : "No");
        printf("  Memory Expander: %s\n", cxlInfo.bMemoryExpander ? "Yes" : "No");
        printf("  CXL Version: %d\n", cxlInfo.cxlVersion);
        printf("  Links: %d (max %d)\n", cxlInfo.nrLinks, cxlInfo.maxNrLinks);
        printf("  Bandwidth: %d MB/s per link\n", cxlInfo.perLinkBwMBps);
        printf("  Remote Type: %d\n", cxlInfo.remoteType);
        g_ctx.cxlVersion = cxlInfo.cxlVersion;
    }

    /* Run P2P DMA tests with different sizes */
    size_t sizes[] = {64*1024, 256*1024, 1*1024*1024, 4*1024*1024, 16*1024*1024};
    for (size_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
        test_p2p_dma(sizes[i]);
    }

    cleanup();
    printf("\n=== Test Complete ===\n");
    return 0;
}
