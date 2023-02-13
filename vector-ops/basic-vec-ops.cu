/**
 * CS61064 - High Performance Parallel Programming - CUDA/GPU
 * Assignment 1(a) - Linear Transformations
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


const double EPS = 1e-6;


__global__
void process_kernel1(const float *input1, const float *input2, float *output,
                     int n)
{
    int block_num = blockIdx.z * (gridDim.y * gridDim.x) \
                  + blockIdx.y * gridDim.x + blockIdx.x;
    int thread_num = threadIdx.z * (blockDim.y * blockDim.x ) \
                   + threadIdx.y * blockDim.x + threadIdx.x;
    int i = block_num * (blockDim.z * blockDim.y * blockDim.x) + thread_num;

    if (i < n) {
        output[i] = sin(input1[i]) + cos(input2[i]);
    }
}


int test_process_kernel1(const float *input1, const float *input2,
                         const float *output, int n)
{
    for (int i = 0; i < n; i++) {
        float expected_output = sin(input1[i]) + cos(input2[i]);
        if (abs(output[i] - expected_output) > EPS) {
            return -1;
        }
    }
    return 0;
}


__global__
void process_kernel2(const float *input, float *output, int n)
{
    int block_num = blockIdx.z * (gridDim.y * gridDim.x) \
                  + blockIdx.y * gridDim.x + blockIdx.x;
    int thread_num = threadIdx.z * (blockDim.y * blockDim.x ) \
                   + threadIdx.y * blockDim.x + threadIdx.x;
    int i = block_num * (blockDim.z * blockDim.y * blockDim.x) + thread_num;

    if (i < n) {
        output[i] = log(input[i]);
    }
}


int test_process_kernel2(const float *input, const float *output, int n)
{
    for (int i = 0; i < n; i++) {
        float expected_output = log(input[i]);
        if (abs(output[i] - expected_output) > EPS) {
            return -1;
        }
    }
    return 0;
}


__global__ 
void process_kernel3(const float *input, float *output, int n)
{
    int block_num = blockIdx.z * (gridDim.y * gridDim.x) \
                  + blockIdx.y * gridDim.x + blockIdx.x;
    int thread_num = threadIdx.z * (blockDim.y * blockDim.x ) \
                   + threadIdx.y * blockDim.x + threadIdx.x;
    int i = block_num * (blockDim.z * blockDim.y * blockDim.x) + thread_num;

    if (i < n) {
        output[i] = sqrt(input[i]);
    }
}


int test_process_kernel3(const float *input, const float *output, int n)
{
    for (int i = 0; i < n; i++) {
        float expected_output = sqrt(input[i]);
        if (abs(output[i] - expected_output) > EPS) {
            return -1;
        }
    }
    return 0;
}


int main(int argc, char *argv[]) 
{
    cudaError_t err = cudaSuccess;
    void *ptr = NULL;
    int ret = 0;
    
    int num_elements = (1 << 14);
    size_t size = num_elements * sizeof(float);

    float *h_A = (float *) malloc(size); CHECK_HOST_ERR(h_A);
    float *h_B = (float *) malloc(size); CHECK_HOST_ERR(h_B);
    float *h_C = (float *) malloc(size); CHECK_HOST_ERR(h_C);

    for (int i = 0; i < num_elements; i++) {
        h_A[i] = rand() / (float) RAND_MAX;
        h_B[i] = rand() / (float) RAND_MAX;
    }

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CHECK_DEVICE_ERR(cudaMalloc((void **)&d_A, size));
    CHECK_DEVICE_ERR(cudaMalloc((void **)&d_B, size));
    CHECK_DEVICE_ERR(cudaMalloc((void **)&d_C, size));
    CHECK_DEVICE_ERR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_DEVICE_ERR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 grid_kernel1(4, 2, 2);
    dim3 block_kernel1(32, 32, 1);
    process_kernel1<<<grid_kernel1, block_kernel1>>>(d_A, d_B, d_C, num_elements);
    CHECK_DEVICE_ERR(cudaGetLastError());
    CHECK_DEVICE_ERR(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    ret = test_process_kernel1(h_A, h_B, h_C, num_elements);
    if (ret) {
        LOG("\tError: Test failed for kernel 1!\n");
    } else {
        LOG("\tSuccess: Test passed for kernel 1.\n");
    }

    dim3 grid_kernel2(2, 8, 1);
    dim3 block_kernel2(8, 8, 16);
    process_kernel2<<<grid_kernel2, block_kernel2>>>(d_C, d_A, num_elements);
    CHECK_DEVICE_ERR(cudaGetLastError());
    CHECK_DEVICE_ERR(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));

    ret = test_process_kernel2(h_C, h_A, num_elements);
    if (ret) {
        LOG("\tError: Test failed for kernel 2!\n");
    } else {
        LOG("\tSuccess: Test passed for kernel 2.\n");
    }

    dim3 grid_kernel3(16, 1, 1);
    dim3 block_kernel3(128, 8, 1);
    process_kernel3<<<grid_kernel3, block_kernel3>>>(d_A, d_B, num_elements);
    CHECK_DEVICE_ERR(cudaGetLastError());
    CHECK_DEVICE_ERR(cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost));

    ret = test_process_kernel3(h_A, h_B, num_elements);
    if (ret) {
        LOG("\tError: Test failed for kernel 3!\n");
    } else {
        LOG("\tSuccess: Test passed for kernel 3.\n");
    }

    return 0;
}