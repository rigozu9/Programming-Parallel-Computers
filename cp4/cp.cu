#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                   \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",         \
                         __FILE__, __LINE__, cudaGetErrorString(err));\
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- the correlation between rows i and j has to be stored in result[i + j*ny]
- only elements with 0 <= j <= i < ny need to be filled
*/

__global__ void normalize_rows_kernel(int ny, int nx, const float* data, float* normalized) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= ny) {
        return;
    }

    float sum = 0;

    for (int x = 0; x < nx; x++) {
        sum += data[x + y*nx];
    }

    float mean = sum / nx;

    float sq_sum = 0;

    for (int x = 0; x < nx; x++) {
        normalized[x + y*nx] = data[x + y*nx] - mean;
        sq_sum += normalized[x + y*nx] * normalized[x + y*nx];
    }

    float length = sqrtf(sq_sum);

    for (int x = 0; x < nx; x++) {
        if (length > 0) {
            normalized[x + y*nx] /= length;
        } else {
            normalized[x + y*nx] = 0;
        }
    }
}

__global__ void correlate_kernel(int ny, int nx, const float* normalized, float* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= ny || j >= ny) {
        return;
    }

    if (j > i) {
        return;
    }

    float dot = 0;

    for (int x = 0; x < nx; x++) {
        dot += normalized[x + i * nx] * normalized[x + j * nx];
    }

    result[i + j * ny] = dot;
}

void correlate(int ny, int nx, const float *data, float *result) {
    float* dataGPU;
    float* normalizedGPU;
    float* resultGPU;

    std::size_t dataSize = ny * nx * sizeof(float);
    std::size_t resultSize = ny * ny * sizeof(float);

    CHECK(cudaMalloc(&dataGPU, dataSize));
    CHECK(cudaMalloc(&normalizedGPU, dataSize));
    CHECK(cudaMalloc(&resultGPU, resultSize));

    // CUDA initcheck fails.
    // Initialize resultGPU so the unused triangle (j > i) is not left uninitialized.
    CHECK(cudaMemset(resultGPU, 0, resultSize));

    CHECK(cudaMemcpy(dataGPU, data, dataSize, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocks = (ny + threadsPerBlock - 1) / threadsPerBlock;

    normalize_rows_kernel<<<blocks, threadsPerBlock>>>(ny, nx, dataGPU, normalizedGPU);
    CHECK(cudaGetLastError());

    dim3 block2(16, 16);
    dim3 grid2((ny + block2.x - 1) / block2.x, (ny + block2.y - 1) / block2.y);

    correlate_kernel<<<grid2, block2>>>(ny, nx, normalizedGPU, resultGPU);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, resultGPU, resultSize, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(dataGPU));
    CHECK(cudaFree(normalizedGPU));
    CHECK(cudaFree(resultGPU));
}
