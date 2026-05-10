#include<cmath>
typedef float float8_t __attribute__ ((vector_size (8 * sizeof(float))));
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

// helper for summing the lanes together
static inline float sum_lanes(float8_t v) {
    return v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7];
}

void correlate(int ny, int nx, const float *data, float *result) {
    int no_of_blocks = (nx + 7) / 8;
    float8_t* normalized = new float8_t[ny * no_of_blocks];

    #pragma omp parallel for
    for (int y = 0; y < ny; y++) {
        float sum = 0;

        for (int x = 0; x < nx; x++) {
            sum += data[x + y*nx];
        }

        float mean = sum / nx;

        float sq_sum = 0;

        for (int x = 0; x < nx; x++) {
            float v = data[x + y*nx] - mean;
            sq_sum += v * v;
        }

        float length = sqrtf(sq_sum);

        for (int k = 0; k < no_of_blocks; k++) {
            float8_t v = {0,0,0,0,0,0,0,0};

            for (int lane = 0; lane < 8; lane++) {
                int x = k * 8 + lane;

                if (x < nx && length > 0) {
                    v[lane] = (data[x + y*nx] - mean) / length;
                } else {
                    v[lane] = 0;
                }
            }

            normalized[y*no_of_blocks + k] = v;
        }
    }
    int end_block = ny - (ny % 4);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i0 = 0; i0 < end_block; i0 += 4) {
        for (int j0 = 0; j0 + 3 < i0; j0 += 4) {
            float8_t sum00 = {0,0,0,0,0,0,0,0};
            float8_t sum01 = {0,0,0,0,0,0,0,0};
            float8_t sum02 = {0,0,0,0,0,0,0,0};
            float8_t sum03 = {0,0,0,0,0,0,0,0};

            float8_t sum10 = {0,0,0,0,0,0,0,0};
            float8_t sum11 = {0,0,0,0,0,0,0,0};
            float8_t sum12 = {0,0,0,0,0,0,0,0};
            float8_t sum13 = {0,0,0,0,0,0,0,0};

            float8_t sum20 = {0,0,0,0,0,0,0,0};
            float8_t sum21 = {0,0,0,0,0,0,0,0};
            float8_t sum22 = {0,0,0,0,0,0,0,0};
            float8_t sum23 = {0,0,0,0,0,0,0,0};

            float8_t sum30 = {0,0,0,0,0,0,0,0};
            float8_t sum31 = {0,0,0,0,0,0,0,0};
            float8_t sum32 = {0,0,0,0,0,0,0,0};
            float8_t sum33 = {0,0,0,0,0,0,0,0};

            for (int k = 0; k < no_of_blocks; k++) {
                float8_t a0 = normalized[(i0+0)*no_of_blocks + k];
                float8_t a1 = normalized[(i0+1)*no_of_blocks + k];
                float8_t a2 = normalized[(i0+2)*no_of_blocks + k];
                float8_t a3 = normalized[(i0+3)*no_of_blocks + k];

                float8_t b0 = normalized[(j0+0)*no_of_blocks + k];
                float8_t b1 = normalized[(j0+1)*no_of_blocks + k];
                float8_t b2 = normalized[(j0+2)*no_of_blocks + k];
                float8_t b3 = normalized[(j0+3)*no_of_blocks + k];

                sum00 += a0 * b0;
                sum01 += a0 * b1;
                sum02 += a0 * b2;
                sum03 += a0 * b3;
                
                sum10 += a1 * b0;
                sum11 += a1 * b1;
                sum12 += a1 * b2;
                sum13 += a1 * b3;

                sum20 += a2 * b0;
                sum21 += a2 * b1;
                sum22 += a2 * b2;
                sum23 += a2 * b3;

                sum30 += a3 * b0;
                sum31 += a3 * b1;
                sum32 += a3 * b2;
                sum33 += a3 * b3;
            }

            result[(i0+0) + (j0+0)*ny] = sum_lanes(sum00);
            result[(i0+0) + (j0+1)*ny] = sum_lanes(sum01);
            result[(i0+0) + (j0+2)*ny] = sum_lanes(sum02);
            result[(i0+0) + (j0+3)*ny] = sum_lanes(sum03);

            result[(i0+1) + (j0+0)*ny] = sum_lanes(sum10);
            result[(i0+1) + (j0+1)*ny] = sum_lanes(sum11);
            result[(i0+1) + (j0+2)*ny] = sum_lanes(sum12);
            result[(i0+1) + (j0+3)*ny] = sum_lanes(sum13);

            result[(i0+2) + (j0+0)*ny] = sum_lanes(sum20);
            result[(i0+2) + (j0+1)*ny] = sum_lanes(sum21);
            result[(i0+2) + (j0+2)*ny] = sum_lanes(sum22);
            result[(i0+2) + (j0+3)*ny] = sum_lanes(sum23);

            result[(i0+3) + (j0+0)*ny] = sum_lanes(sum30);
            result[(i0+3) + (j0+1)*ny] = sum_lanes(sum31);
            result[(i0+3) + (j0+2)*ny] = sum_lanes(sum32);
            result[(i0+3) + (j0+3)*ny] = sum_lanes(sum33);
        }

        // #pragma omp parallel for schedule(dynamic, 1)
        for (int ii = i0; ii <= i0 + 3; ii++) {
            for (int j = i0; j <= ii; j++) {
                float8_t dot8 = {0,0,0,0,0,0,0,0};

                for (int k = 0; k < no_of_blocks; k++) {
                    float8_t a = normalized[ii*no_of_blocks + k];
                    float8_t b = normalized[j*no_of_blocks + k];
                    dot8 += a * b;
                }

                result[ii + j*ny] = sum_lanes(dot8);
            }
        }
    }
    #pragma omp parallel for schedule(dynamic, 1)
    for (int ii = end_block; ii < ny; ii++) {
        for (int j = 0; j <= ii; j++) {
            float8_t dot8 = {0,0,0,0,0,0,0,0};

            for (int k = 0; k < no_of_blocks; k++) {
                float8_t a = normalized[ii*no_of_blocks + k];
                float8_t b = normalized[j*no_of_blocks + k];
                dot8 += a * b;
            }

            result[ii + j*ny] = sum_lanes(dot8);
        }
    }
    delete[] normalized;
}