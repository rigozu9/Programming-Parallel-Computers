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

#define TILE 16
#define CHUNK 64
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
    int i0 = blockIdx.x * TILE;
    int j0 = blockIdx.y * TILE;

    if (j0 > i0 + TILE - 1) {
        return;
    }

    int i = i0 + threadIdx.x;
    int j = j0 + threadIdx.y;

    bool valid = (i < ny && j < ny && j <= i);

    float dot = 0;

    __shared__ float tile_i[TILE][CHUNK];
    __shared__ float tile_j[TILE][CHUNK];

    int tid = threadIdx.y * TILE + threadIdx.x;

    for (int x0 = 0; x0 < nx; x0 += CHUNK) {
        for (int t = tid; t < TILE * CHUNK; t += TILE * TILE) {
            int row = t / CHUNK;
            int col = t % CHUNK;

            int x = x0 + col;

            if (i0 + row < ny && x < nx) {
                tile_i[row][col] = normalized[x + (i0 + row) * nx];
            } else {
                tile_i[row][col] = 0;
            }

            if (j0 + row < ny && x < nx) {
                tile_j[row][col] = normalized[x + (j0 + row) * nx];
            } else {
                tile_j[row][col] = 0;
            }
        }

        __syncthreads();

        if (valid) {
            for (int k = 0; k < CHUNK; k++) {
                dot += tile_i[threadIdx.x][k] * tile_j[threadIdx.y][k];
            }
        }

        __syncthreads();
    }

    if (valid) {
        result[i + j * ny] = dot;
    }
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
