/**
 * CS61064 - High Performance Parallel Programming - CUDA/GPU
 * Assignment 1(b) - Convolutions
 * 
 * Author: Utkarsh Patel (18EC35034)
 */

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define LOG(...) \
    fprintf(stdout, "  File \"%s\", Function \"%s\", line %d\n", \
        __FILE__, __PRETTY_FUNCTION__, __LINE__); \
    fprintf(stdout, __VA_ARGS__); 

#define CHECK_HOST_ERR(...) \
    ptr = __VA_ARGS__; \
    if (ptr == NULL) { \
        fprintf(stderr, "  File \"%s\", Function \"%s\", line %d\n", \
            __FILE__, __PRETTY_FUNCTION__, __LINE__); \
        fprintf(stderr, "\tNull pointer exception!\n"); \
        exit(EXIT_FAILURE); \
    }

#define CHECK_DEVICE_ERR(...) \
    err = __VA_ARGS__; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "  File \"%s\", Function \"%s\", line %d\n", \
            __FILE__, __PRETTY_FUNCTION__, __LINE__); \
        fprintf(stderr, "\t%s: %s\n", cudaGetErrorName(err), \
            cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }


#define DIM_1_KERNEL_X 5
#define DIM_1_PAD_X DIM_1_KERNEL_X / 2

__constant__ float DIM_1_KERNEL[DIM_1_KERNEL_X];

#define DIM_2_KERNEL_X 3
#define DIM_2_KERNEL_Y 3
#define DIM_2_PAD_X DIM_2_KERNEL_X / 2
#define DIM_2_PAD_Y DIM_2_KERNEL_Y / 2

__constant__ float DIM_2_KERNEL[DIM_2_KERNEL_X * DIM_2_KERNEL_Y];

#define DIM_3_KERNEL_X 3
#define DIM_3_KERNEL_Y 3
#define DIM_3_KERNEL_Z 3
#define DIM_3_PAD_X DIM_3_KERNEL_X / 2
#define DIM_3_PAD_Y DIM_3_KERNEL_Y / 2
#define DIM_3_PAD_Z DIM_3_KERNEL_Z / 2

__constant__ float DIM_3_KERNEL[DIM_3_KERNEL_X * DIM_3_KERNEL_Y * DIM_3_KERNEL_Z];

const double EPS = 1e-6;

__device__ __host__
inline float conv1d_fn(float a, float b)
{
    return max(a, b);
}

__global__
void Conv1D(const float *padded_input, float *output, int n)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if (x < n) {
        float conv = -INFINITY, cur;
        for (int dx = 0; dx < DIM_1_KERNEL_X; dx++) {
            cur = padded_input[x + dx] * DIM_1_KERNEL[dx];
            conv = conv1d_fn(conv, cur);
        }
        output[x] = conv;
    }
}

int test_Conv1D(const float *padded_input, const float *output, 
                const float *kernel, int n)
{
    for (int x = 0; x < n; x++) {
        float conv = -INFINITY, cur;
        for (int dx = 0; dx < DIM_1_KERNEL_X; dx++) {
            cur = padded_input[x + dx] * kernel[dx];
            conv = conv1d_fn(conv, cur);
        }
        if (abs(output[x] - conv) > EPS) {
            return -1;
        }
    }
    return 0;
}


__device__ __host__
inline float conv2d_fn(float a, float b)
{
    return a + b;
}


__global__ 
void Conv2D(const float *padded_input, float *output, int rows, int cols)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((x < cols) && (y < rows)) {
        float conv = 0, cur;
        for (int dy = 0; dy < DIM_2_KERNEL_Y; dy++) {
            for (int dx = 0; dx < DIM_2_KERNEL_X; dx++) {
                int input_idx = (y + dy) * (cols + DIM_2_PAD_X * 2) + x + dx;
                int kernel_idx = dy * DIM_2_KERNEL_X + dx;
                cur = padded_input[input_idx] * DIM_2_KERNEL[kernel_idx];
                conv = conv2d_fn(conv, cur);
            }
        }
        int output_idx = y * cols + x;
        output[output_idx] = conv;
    }
}

int test_Conv2D(const float *padded_input, const float *output, 
                const float *kernel, int rows, int cols)
{
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            float conv = 0, cur;
            for (int dy = 0; dy < DIM_2_KERNEL_Y; dy++) {
                for (int dx = 0; dx < DIM_2_KERNEL_X; dx++) {
                    int input_idx = (y + dy) * (cols + DIM_2_PAD_X * 2) + x + dx;
                    int kernel_idx = dy * DIM_2_KERNEL_X + dx;
                    cur = padded_input[input_idx] * kernel[kernel_idx];
                    conv = conv2d_fn(conv, cur);
                }
            }
            int output_idx = y * cols + x;
            if (abs(output[output_idx] - conv) > EPS) {
                return -1;
            }
        }
    }
    return 0;
}

__device__ __host__
inline float conv3d_fn(float a, float b)
{
    return a + b;
}

__global__
void Conv3D(const float *padded_input, float *output, int depth, int rows, int cols)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if ((x < cols) && (y < rows) && (z < depth)) {
        float conv = 0, cur;
        for (int dz = 0; dz < DIM_3_KERNEL_Z; dz++) {
            for (int dy = 0; dy < DIM_3_KERNEL_Y; dy++) {
                for (int dx = 0; dx < DIM_3_KERNEL_X; dx++) {
                    int input_idx = (z + dz) * ((rows + DIM_3_PAD_Y * 2) * \
                        (cols + DIM_3_PAD_X * 2)) + (y + dy) * \
                        (cols + DIM_3_PAD_X * 2) + x + dx;
                    int kernel_idx = dz * DIM_3_KERNEL_Y * DIM_3_KERNEL_X + \
                        dy * DIM_3_KERNEL_X + dx;
                    cur = padded_input[input_idx] * DIM_3_KERNEL[kernel_idx];
                    conv = conv3d_fn(conv, cur);
                }
            }
        }
        int output_idx = z * rows * cols + y * cols + x;
        output[output_idx] = conv;
    }
}


int test_Conv3D(const float *padded_input, const float *output,
                const float *kernel, int depth, int rows, int cols)
{
    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                float conv = 0, cur;
                for (int dz = 0; dz < DIM_3_KERNEL_Z; dz++) {
                    for (int dy = 0; dy < DIM_3_KERNEL_Y; dy++) {
                        for (int dx = 0; dx < DIM_3_KERNEL_X; dx++) {
                            int input_idx = (z + dz) * ((rows + DIM_3_PAD_Y * 2) * \
                                (cols + DIM_3_PAD_X * 2)) + (y + dy) * \
                                (cols + DIM_3_PAD_X * 2) + x + dx;
                            int kernel_idx = dz * DIM_3_KERNEL_Y * DIM_3_KERNEL_X + \
                                dy * DIM_3_KERNEL_X + dx;
                            cur = padded_input[input_idx] * kernel[kernel_idx];
                            conv = conv3d_fn(conv, cur);
                        }
                    }
                }
                int output_idx = z * rows * cols + y * cols + x;
                if (abs(output[output_idx] - conv) > EPS) {
                    return -1;
                }
            }
        }
    }
    return 0;
}


int main(int argc, char *argv[])
{
    cudaError_t err = cudaSuccess;
    void *ptr = NULL;
    int ret = 0, num_elements, rows, cols, depth;
    size_t size_A, size_B, size_K;
    float *h_A, *h_B, *h_K, *d_A, *d_B;
    int n_threads_x, n_threads_y, n_threads_z;
    int n_blocks_x, n_blocks_y, n_blocks_z;
    cudaEvent_t start1d, start2d, start3d, stop1d, stop2d, stop3d;
    float elapsed_time;

    /*=========================== CONV 1D starts =============================*/

    cudaEventCreate(&start1d);
    cudaEventCreate(&stop1d);

    num_elements = (1 << 20);

    size_A = (num_elements + DIM_1_PAD_X * 2) * sizeof(float);
    size_B = num_elements * sizeof(float);
    size_K = DIM_1_KERNEL_X * sizeof(float);

    h_A = (float *) malloc(size_A); CHECK_HOST_ERR(h_A);
    h_B = (float *) malloc(size_B); CHECK_HOST_ERR(h_B);
    h_K = (float *) malloc(size_K); CHECK_HOST_ERR(h_K);
    memset(h_A, 0, size_A);

    for (int x = DIM_1_PAD_X; x < num_elements + DIM_1_PAD_X; x++) {
        h_A[x] = rand() / (float) RAND_MAX;
    }

    for (int x = -DIM_1_PAD_X; x <= DIM_1_PAD_X; x++) {
        h_K[x + DIM_1_PAD_X] = (float) (x != 0);
    }

    CHECK_DEVICE_ERR(cudaMalloc((void **)&d_A, size_A));
    CHECK_DEVICE_ERR(cudaMalloc((void **)&d_B, size_B));
    CHECK_DEVICE_ERR(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_DEVICE_ERR(cudaMemcpyToSymbol(DIM_1_KERNEL, h_K, size_K));

    n_threads_x = 1024;
    n_blocks_x  = ceil(num_elements / (double) n_threads_x);

    dim3 grid_conv1d(n_blocks_x, 1, 1);
    dim3 block_conv1d(n_threads_x, 1, 1);
    cudaEventRecord(start1d);
    Conv1D<<<grid_conv1d, block_conv1d>>>(d_A, d_B, num_elements);
    cudaEventRecord(stop1d);

    CHECK_DEVICE_ERR(cudaGetLastError());
    CHECK_DEVICE_ERR(cudaMemcpy(h_B, d_B, size_B, cudaMemcpyDeviceToHost));

    ret = test_Conv1D(h_A, h_B, h_K, num_elements);
    if (ret) {
        LOG("\tError: Test failed for Conv1D!\n");
    } else {
        LOG("\tSuccess: Test passed for Conv1D.\n");
    }

    cudaEventSynchronize(stop1d);
    cudaEventElapsedTime(&elapsed_time, start1d, stop1d);
    LOG("\tConv1D: %.4f µs\n", elapsed_time * 1000);

    CHECK_DEVICE_ERR(cudaFree(d_A));
    CHECK_DEVICE_ERR(cudaFree(d_B));
    free(h_A); free(h_B); free(h_K);

    /*============================ CONV 1D ends ==============================*/

    /*=========================== CONV 2D starts =============================*/

    cudaEventCreate(&start2d);
    cudaEventCreate(&stop2d);

    rows = (1 << 10);
    cols = (1 << 10);

    size_A = (cols + DIM_2_PAD_X * 2) * (rows + DIM_2_PAD_Y * 2) * sizeof(float);
    size_B = cols * rows * sizeof(float);
    size_K = DIM_2_KERNEL_X * DIM_2_KERNEL_Y * sizeof(float);

    h_A = (float *) malloc(size_A); CHECK_HOST_ERR(h_A);
    h_B = (float *) malloc(size_B); CHECK_HOST_ERR(h_B);
    h_K = (float *) malloc(size_K); CHECK_HOST_ERR(h_K);
    memset(h_A, 0, size_A);

    for (int y = DIM_2_PAD_Y; y < rows + DIM_2_PAD_Y; y++) {
        for (int x = DIM_2_PAD_X; x < cols + DIM_2_PAD_X; x++) {
            int idx = y * (cols + DIM_2_PAD_X * 2) + x;
            h_A[idx] = rand() / (float) RAND_MAX;
        }
    }

    for (int y = -DIM_2_PAD_Y; y <= DIM_2_PAD_Y; y++) {
        for (int x = -DIM_2_PAD_X; x <= DIM_2_PAD_X; x++) {
            int idx = (y + DIM_2_PAD_Y) * DIM_2_KERNEL_X + (x + DIM_2_PAD_X);
            h_K[idx] = (float)(x != 0 || y != 0) / 8.0;
        }
    }

    CHECK_DEVICE_ERR(cudaMalloc((void **)&d_A, size_A));
    CHECK_DEVICE_ERR(cudaMalloc((void **)&d_B, size_B));
    CHECK_DEVICE_ERR(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_DEVICE_ERR(cudaMemcpyToSymbol(DIM_2_KERNEL, h_K, size_K));

    n_threads_x = 32;
    n_threads_y = 32;
    n_blocks_x  = ceil(cols / (double) n_threads_x);
    n_blocks_y  = ceil(rows / (double) n_threads_y);

    dim3 grid_conv2d(n_blocks_x, n_blocks_y, 1);
    dim3 block_conv2d(n_threads_x, n_threads_y, 1);
    cudaEventRecord(start2d);
    Conv2D<<<grid_conv2d, block_conv2d>>>(d_A, d_B, rows, cols);
    cudaEventRecord(stop2d);

    CHECK_DEVICE_ERR(cudaGetLastError());
    CHECK_DEVICE_ERR(cudaMemcpy(h_B, d_B, size_B, cudaMemcpyDeviceToHost));

    ret = test_Conv2D(h_A, h_B, h_K, rows, cols);
    if (ret) {
        LOG("\tError: Test failed for Conv2D!\n");
    } else {
        LOG("\tSuccess: Test passed for Conv2D.\n");
    }

    cudaEventSynchronize(stop2d);
    cudaEventElapsedTime(&elapsed_time, start2d, stop2d);
    LOG("\tConv2D: %.4f µs\n", elapsed_time * 1000);

    CHECK_DEVICE_ERR(cudaFree(d_A));
    CHECK_DEVICE_ERR(cudaFree(d_B));
    free(h_A); free(h_B); free(h_K);

    /*============================ CONV 2D ends ==============================*/

    /*=========================== CONV 3D starts =============================*/

    cudaEventCreate(&start3d);
    cudaEventCreate(&stop3d);

    rows  = (1 << 6);
    cols  = (1 << 6);
    depth = (1 << 6);

    size_A = (cols + DIM_3_PAD_X * 2) * (rows + DIM_3_PAD_Y * 2) \
             * (depth + DIM_3_PAD_Z * 2) * sizeof(float);
    size_B = cols * rows * depth * sizeof(float);
    size_K = DIM_3_KERNEL_X * DIM_3_KERNEL_Y * DIM_3_KERNEL_Z * sizeof(float);

    h_A = (float *) malloc(size_A); CHECK_HOST_ERR(h_A);
    h_B = (float *) malloc(size_B); CHECK_HOST_ERR(h_B);
    h_K = (float *) malloc(size_K); CHECK_HOST_ERR(h_K);
    memset(h_A, 0, size_A);

    for (int z = DIM_3_PAD_Z; z < depth + DIM_3_PAD_Z; z++) {
        for (int y = DIM_3_PAD_Y; y < rows + DIM_3_PAD_Y; y++) {
            for (int x = DIM_3_PAD_X; x < cols + DIM_3_PAD_X; x++) {
                int idx = z * (cols + DIM_3_PAD_X * 2) * (rows + DIM_3_PAD_Y * 2) \
                          + y * (cols + DIM_3_PAD_X * 2) + x;
                h_A[idx] = rand() / (float) RAND_MAX;
            }
        }
    }

    for (int z = -DIM_3_PAD_Z; z <= DIM_3_PAD_Z; z++) {
        for (int y = -DIM_3_PAD_Y; y <= DIM_3_PAD_Y; y++) {
            for (int x = -DIM_3_PAD_X; x <= DIM_3_PAD_X; x++) {
                int idx = (z + DIM_3_PAD_Z) * DIM_3_KERNEL_X * DIM_3_KERNEL_Y \
                          + (y + DIM_3_PAD_Y) * DIM_3_KERNEL_X + x + DIM_3_PAD_X;
                h_K[idx] = (float)(x != 0 || y != 0);
            }
        }
    }

    CHECK_DEVICE_ERR(cudaMalloc((void **)&d_A, size_A));
    CHECK_DEVICE_ERR(cudaMalloc((void **)&d_B, size_B));
    CHECK_DEVICE_ERR(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_DEVICE_ERR(cudaMemcpyToSymbol(DIM_3_KERNEL, h_K, size_K));

    n_threads_x = 16;
    n_threads_y = 8;
    n_threads_z = 8;
    n_blocks_x  = ceil(cols  / (double) n_threads_x);
    n_blocks_y  = ceil(rows  / (double) n_threads_y);
    n_blocks_z  = ceil(depth / (double) n_threads_z);

    dim3 grid_conv3d(n_blocks_x, n_blocks_y, n_blocks_z);
    dim3 block_conv3d(n_threads_x, n_threads_y, n_threads_z);
    cudaEventRecord(start3d);
    Conv3D<<<grid_conv3d, block_conv3d>>>(d_A, d_B, depth, rows, cols);
    cudaEventRecord(stop3d);

    CHECK_DEVICE_ERR(cudaGetLastError());
    CHECK_DEVICE_ERR(cudaMemcpy(h_B, d_B, size_B, cudaMemcpyDeviceToHost));

    ret = test_Conv3D(h_A, h_B, h_K, depth, rows, cols);
    if (ret) {
        LOG("\tError: Test failed for Conv3D!\n");
    } else {
        LOG("\tSuccess: Test passed for Conv3D.\n");
    }

    cudaEventSynchronize(stop3d);
    cudaEventElapsedTime(&elapsed_time, start3d, stop3d);
    LOG("\tConv3D: %.4f µs\n", elapsed_time * 1000);

    CHECK_DEVICE_ERR(cudaFree(d_A));
    CHECK_DEVICE_ERR(cudaFree(d_B));
    free(h_A); free(h_B); free(h_K);

    return 0; 
}
