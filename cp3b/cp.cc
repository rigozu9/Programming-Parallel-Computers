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
    int end_block = ny - (ny % 8);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i0 = 0; i0 < end_block; i0 += 8) {
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

            float8_t sum40 = {0,0,0,0,0,0,0,0};
            float8_t sum41 = {0,0,0,0,0,0,0,0};
            float8_t sum42 = {0,0,0,0,0,0,0,0};
            float8_t sum43 = {0,0,0,0,0,0,0,0};

            float8_t sum50 = {0,0,0,0,0,0,0,0};
            float8_t sum51 = {0,0,0,0,0,0,0,0};
            float8_t sum52 = {0,0,0,0,0,0,0,0};
            float8_t sum53 = {0,0,0,0,0,0,0,0};

            float8_t sum60 = {0,0,0,0,0,0,0,0};
            float8_t sum61 = {0,0,0,0,0,0,0,0};
            float8_t sum62 = {0,0,0,0,0,0,0,0};
            float8_t sum63 = {0,0,0,0,0,0,0,0};

            float8_t sum70 = {0,0,0,0,0,0,0,0};
            float8_t sum71 = {0,0,0,0,0,0,0,0};
            float8_t sum72 = {0,0,0,0,0,0,0,0};
            float8_t sum73 = {0,0,0,0,0,0,0,0};

            for (int k = 0; k < no_of_blocks; k++) {
                float8_t a0 = normalized[(i0+0)*no_of_blocks + k];
                float8_t a1 = normalized[(i0+1)*no_of_blocks + k];
                float8_t a2 = normalized[(i0+2)*no_of_blocks + k];
                float8_t a3 = normalized[(i0+3)*no_of_blocks + k];
                float8_t a4 = normalized[(i0+4)*no_of_blocks + k];
                float8_t a5 = normalized[(i0+5)*no_of_blocks + k];
                float8_t a6 = normalized[(i0+6)*no_of_blocks + k];
                float8_t a7 = normalized[(i0+7)*no_of_blocks + k];

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

                sum40 += a4 * b0;
                sum41 += a4 * b1;
                sum42 += a4 * b2;
                sum43 += a4 * b3;

                sum50 += a5 * b0;
                sum51 += a5 * b1;
                sum52 += a5 * b2;
                sum53 += a5 * b3;

                sum60 += a6 * b0;
                sum61 += a6 * b1;
                sum62 += a6 * b2;
                sum63 += a6 * b3;

                sum70 += a7 * b0;
                sum71 += a7 * b1;
                sum72 += a7 * b2;
                sum73 += a7 * b3;
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

            result[(i0+4) + (j0+0)*ny] = sum_lanes(sum40);
            result[(i0+4) + (j0+1)*ny] = sum_lanes(sum41);
            result[(i0+4) + (j0+2)*ny] = sum_lanes(sum42);
            result[(i0+4) + (j0+3)*ny] = sum_lanes(sum43);

            result[(i0+5) + (j0+0)*ny] = sum_lanes(sum50);
            result[(i0+5) + (j0+1)*ny] = sum_lanes(sum51);
            result[(i0+5) + (j0+2)*ny] = sum_lanes(sum52);
            result[(i0+5) + (j0+3)*ny] = sum_lanes(sum53);

            result[(i0+6) + (j0+0)*ny] = sum_lanes(sum60);
            result[(i0+6) + (j0+1)*ny] = sum_lanes(sum61);
            result[(i0+6) + (j0+2)*ny] = sum_lanes(sum62);
            result[(i0+6) + (j0+3)*ny] = sum_lanes(sum63);

            result[(i0+7) + (j0+0)*ny] = sum_lanes(sum70);
            result[(i0+7) + (j0+1)*ny] = sum_lanes(sum71);
            result[(i0+7) + (j0+2)*ny] = sum_lanes(sum72);
            result[(i0+7) + (j0+3)*ny] = sum_lanes(sum73);
        }

        // #pragma omp parallel for schedule(dynamic, 1)
        for (int ii = i0; ii <= i0 + 7; ii++) {
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