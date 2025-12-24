/*
 * NVSHMEM CXL Transport Integration Test
 *
 * Tests NVSHMEM symmetric heap operations with CXL transport enabled.
 * This test verifies that GPU (Type 2) can communicate with CXL memory (Type 3)
 * using the NVSHMEM API.
 *
 * Build:
 *   Included in NVSHMEM build when NVSHMEM_BUILD_TESTS=ON
 *
 * Run:
 *   export NVSHMEM_ENABLE_CXL_TRANSPORT=1
 *   mpirun -np 2 ./cxl_nvshmem_test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define NVSHMEM_CHECK(call) do { \
    int err = call; \
    if (err != 0) { \
        fprintf(stderr, "NVSHMEM error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while(0)

/* Simple put kernel */
__global__ void put_kernel(int *dest, int *src, int pe, size_t nelems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nelems) {
        nvshmem_int_p(dest + idx, src[idx], pe);
    }
}

/* Simple get kernel */
__global__ void get_kernel(int *dest, int *src, int pe, size_t nelems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nelems) {
        dest[idx] = nvshmem_int_g(src + idx, pe);
    }
}

/* Verification kernel */
__global__ void verify_kernel(int *data, int expected_base, size_t nelems, int *errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nelems) {
        int expected = expected_base + idx;
        if (data[idx] != expected) {
            atomicAdd(errors, 1);
        }
    }
}

void test_put_get(int mype, int npes, size_t nelems) {
    printf("[PE %d] Testing put/get with %zu elements\n", mype, nelems);

    /* Allocate symmetric memory */
    int *shmem_buf = (int *)nvshmem_malloc(nelems * sizeof(int));
    if (!shmem_buf) {
        fprintf(stderr, "[PE %d] nvshmem_malloc failed\n", mype);
        return;
    }

    /* Allocate local GPU buffers */
    int *local_src, *local_dst;
    CUDA_CHECK(cudaMalloc(&local_src, nelems * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&local_dst, nelems * sizeof(int)));

    /* Initialize local source with PE-specific pattern */
    int *host_init = (int *)malloc(nelems * sizeof(int));
    for (size_t i = 0; i < nelems; i++) {
        host_init[i] = mype * 1000 + i;
    }
    CUDA_CHECK(cudaMemcpy(local_src, host_init, nelems * sizeof(int), cudaMemcpyHostToDevice));

    /* Clear destination and symmetric buffer */
    CUDA_CHECK(cudaMemset(local_dst, 0, nelems * sizeof(int)));
    CUDA_CHECK(cudaMemset(shmem_buf, 0, nelems * sizeof(int)));

    nvshmem_barrier_all();

    /* Test PUT: write to next PE's symmetric buffer */
    int target_pe = (mype + 1) % npes;
    int blocks = (nelems + 255) / 256;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    put_kernel<<<blocks, 256>>>(shmem_buf, local_src, target_pe, nelems);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float put_time;
    CUDA_CHECK(cudaEventElapsedTime(&put_time, start, stop));

    nvshmem_barrier_all();

    /* Verify PUT: check our symmetric buffer was written by previous PE */
    int source_pe = (mype + npes - 1) % npes;
    int *errors;
    CUDA_CHECK(cudaMallocManaged(&errors, sizeof(int)));
    *errors = 0;

    int expected_base = source_pe * 1000;
    verify_kernel<<<blocks, 256>>>(shmem_buf, expected_base, nelems, errors);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (*errors > 0) {
        printf("[PE %d] PUT verification FAILED: %d errors\n", mype, *errors);
    } else {
        double bw = (nelems * sizeof(int)) / (put_time * 1e-3) / (1024*1024*1024);
        printf("[PE %d] PUT verification PASSED (%.3f ms, %.2f GB/s)\n", mype, put_time, bw);
    }

    nvshmem_barrier_all();

    /* Test GET: read from previous PE's symmetric buffer */
    *errors = 0;
    CUDA_CHECK(cudaMemset(local_dst, 0, nelems * sizeof(int)));

    CUDA_CHECK(cudaEventRecord(start));
    get_kernel<<<blocks, 256>>>(local_dst, shmem_buf, source_pe, nelems);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float get_time;
    CUDA_CHECK(cudaEventElapsedTime(&get_time, start, stop));

    /* Verify GET */
    int *host_verify = (int *)malloc(nelems * sizeof(int));
    CUDA_CHECK(cudaMemcpy(host_verify, local_dst, nelems * sizeof(int), cudaMemcpyDeviceToHost));

    int get_errors = 0;
    int prev_source = (source_pe + npes - 1) % npes;
    for (size_t i = 0; i < nelems; i++) {
        int expected = prev_source * 1000 + i;
        if (host_verify[i] != expected) {
            get_errors++;
            if (get_errors <= 5) {
                printf("[PE %d] GET mismatch at %zu: expected %d, got %d\n",
                       mype, i, expected, host_verify[i]);
            }
        }
    }

    if (get_errors > 0) {
        printf("[PE %d] GET verification FAILED: %d errors\n", mype, get_errors);
    } else {
        double bw = (nelems * sizeof(int)) / (get_time * 1e-3) / (1024*1024*1024);
        printf("[PE %d] GET verification PASSED (%.3f ms, %.2f GB/s)\n", mype, get_time, bw);
    }

    /* Cleanup */
    free(host_init);
    free(host_verify);
    CUDA_CHECK(cudaFree(errors));
    CUDA_CHECK(cudaFree(local_src));
    CUDA_CHECK(cudaFree(local_dst));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    nvshmem_free(shmem_buf);
}

void test_atomic_add(int mype, int npes) {
    printf("[PE %d] Testing atomic add\n", mype);

    /* Allocate symmetric counter */
    long *counter = (long *)nvshmem_malloc(sizeof(long));
    if (!counter) {
        fprintf(stderr, "[PE %d] nvshmem_malloc failed for atomic test\n", mype);
        return;
    }

    /* Initialize counter on PE 0 */
    if (mype == 0) {
        CUDA_CHECK(cudaMemset(counter, 0, sizeof(long)));
    }

    nvshmem_barrier_all();

    /* Each PE atomically increments counter on PE 0 */
    nvshmem_long_atomic_add(counter, 1, 0);

    nvshmem_barrier_all();

    /* PE 0 verifies the result */
    if (mype == 0) {
        long result;
        CUDA_CHECK(cudaMemcpy(&result, counter, sizeof(long), cudaMemcpyDeviceToHost));
        if (result == npes) {
            printf("[PE %d] Atomic add PASSED: counter = %ld (expected %d)\n",
                   mype, result, npes);
        } else {
            printf("[PE %d] Atomic add FAILED: counter = %ld (expected %d)\n",
                   mype, result, npes);
        }
    }

    nvshmem_free(counter);
}

void test_broadcast(int mype, int npes) {
    printf("[PE %d] Testing broadcast\n", mype);

    size_t nelems = 1024;
    int *data = (int *)nvshmem_malloc(nelems * sizeof(int));
    if (!data) {
        fprintf(stderr, "[PE %d] nvshmem_malloc failed for broadcast test\n", mype);
        return;
    }

    /* Root PE initializes data */
    if (mype == 0) {
        int *host_data = (int *)malloc(nelems * sizeof(int));
        for (size_t i = 0; i < nelems; i++) {
            host_data[i] = 0xDEAD0000 + i;
        }
        CUDA_CHECK(cudaMemcpy(data, host_data, nelems * sizeof(int), cudaMemcpyHostToDevice));
        free(host_data);
    } else {
        CUDA_CHECK(cudaMemset(data, 0, nelems * sizeof(int)));
    }

    nvshmem_barrier_all();

    /* Broadcast from PE 0 to all PEs */
    nvshmem_int_broadcast(NVSHMEM_TEAM_WORLD, data, data, nelems, 0);

    nvshmem_barrier_all();

    /* Verify */
    int *host_verify = (int *)malloc(nelems * sizeof(int));
    CUDA_CHECK(cudaMemcpy(host_verify, data, nelems * sizeof(int), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (size_t i = 0; i < nelems; i++) {
        int expected = 0xDEAD0000 + i;
        if (host_verify[i] != expected) {
            errors++;
        }
    }

    if (errors > 0) {
        printf("[PE %d] Broadcast verification FAILED: %d errors\n", mype, errors);
    } else {
        printf("[PE %d] Broadcast verification PASSED\n", mype);
    }

    free(host_verify);
    nvshmem_free(data);
}

int main(int argc, char *argv[]) {
    /* Initialize NVSHMEM */
    nvshmem_init();

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    /* Set CUDA device based on PE */
    int ndevices;
    CUDA_CHECK(cudaGetDeviceCount(&ndevices));
    CUDA_CHECK(cudaSetDevice(mype % ndevices));

    printf("[PE %d/%d] NVSHMEM CXL Transport Test Started\n", mype, npes);
    printf("[PE %d] Using CUDA device %d\n", mype, mype % ndevices);

    /* Check if CXL transport is enabled */
    const char *cxl_enabled = getenv("NVSHMEM_ENABLE_CXL_TRANSPORT");
    if (cxl_enabled && (strcmp(cxl_enabled, "1") == 0 || strcasecmp(cxl_enabled, "true") == 0)) {
        printf("[PE %d] CXL transport is ENABLED\n", mype);
    } else {
        printf("[PE %d] CXL transport is DISABLED (set NVSHMEM_ENABLE_CXL_TRANSPORT=1 to enable)\n", mype);
    }

    nvshmem_barrier_all();

    /* Run tests with different sizes */
    size_t test_sizes[] = {1024, 16384, 65536, 262144};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        if (mype == 0) {
            printf("\n=== Test with %zu elements (%zu bytes) ===\n",
                   test_sizes[i], test_sizes[i] * sizeof(int));
        }
        nvshmem_barrier_all();
        test_put_get(mype, npes, test_sizes[i]);
        nvshmem_barrier_all();
    }

    /* Test atomics */
    if (mype == 0) printf("\n=== Atomic Operations Test ===\n");
    nvshmem_barrier_all();
    test_atomic_add(mype, npes);
    nvshmem_barrier_all();

    /* Test collective */
    if (mype == 0) printf("\n=== Collective Operations Test ===\n");
    nvshmem_barrier_all();
    test_broadcast(mype, npes);
    nvshmem_barrier_all();

    if (mype == 0) {
        printf("\n=== All Tests Completed ===\n");
    }

    nvshmem_finalize();
    return 0;
}
