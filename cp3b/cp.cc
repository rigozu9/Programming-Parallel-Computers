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
void correlate(int ny, int nx, const float *data, float *result) {
    float* normalized = new float[ny * nx];

    #pragma omp parallel for
    for (int y = 0; y < ny; y++) {
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

        float length = sqrt(sq_sum);

        for (int x = 0; x < nx; x++) {
            if (length > 0) {
                normalized[x + y*nx] /= length;
            } else {
                normalized[x + y*nx] = 0;
            }
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

            float rest00 = 0;
            float rest01 = 0;
            float rest02 = 0; 
            float rest03 = 0;

            float rest10 = 0; 
            float rest11 = 0; 
            float rest12 = 0; 
            float rest13 = 0;

            float rest20 = 0; 
            float rest21 = 0;
            float rest22 = 0;
            float rest23 = 0;

            float rest30 = 0; 
            float rest31 = 0; 
            float rest32 = 0; 
            float rest33 = 0;
            
            int x = 0;

            for (; x + 7 < nx; x+= 8) {
                float8_t a0 = {
                    normalized[(x+0) + (i0+0)*nx],
                    normalized[(x+1) + (i0+0)*nx],
                    normalized[(x+2) + (i0+0)*nx],
                    normalized[(x+3) + (i0+0)*nx],
                    normalized[(x+4) + (i0+0)*nx],
                    normalized[(x+5) + (i0+0)*nx],
                    normalized[(x+6) + (i0+0)*nx],
                    normalized[(x+7) + (i0+0)*nx]
                };

                float8_t a1 = {
                    normalized[(x+0) + (i0+1)*nx],
                    normalized[(x+1) + (i0+1)*nx],
                    normalized[(x+2) + (i0+1)*nx],
                    normalized[(x+3) + (i0+1)*nx],
                    normalized[(x+4) + (i0+1)*nx],
                    normalized[(x+5) + (i0+1)*nx],
                    normalized[(x+6) + (i0+1)*nx],
                    normalized[(x+7) + (i0+1)*nx]
                };

                float8_t a2 = {
                    normalized[(x+0) + (i0+2)*nx],
                    normalized[(x+1) + (i0+2)*nx],
                    normalized[(x+2) + (i0+2)*nx],
                    normalized[(x+3) + (i0+2)*nx],
                    normalized[(x+4) + (i0+2)*nx],
                    normalized[(x+5) + (i0+2)*nx],
                    normalized[(x+6) + (i0+2)*nx],
                    normalized[(x+7) + (i0+2)*nx]
                };

                float8_t a3 = {
                    normalized[(x+0) + (i0+3)*nx],
                    normalized[(x+1) + (i0+3)*nx],
                    normalized[(x+2) + (i0+3)*nx],
                    normalized[(x+3) + (i0+3)*nx],
                    normalized[(x+4) + (i0+3)*nx],
                    normalized[(x+5) + (i0+3)*nx],
                    normalized[(x+6) + (i0+3)*nx],
                    normalized[(x+7) + (i0+3)*nx]
                };

                float8_t b0 = {
                    normalized[(x+0) + (j0+0)*nx],
                    normalized[(x+1) + (j0+0)*nx],
                    normalized[(x+2) + (j0+0)*nx],
                    normalized[(x+3) + (j0+0)*nx],
                    normalized[(x+4) + (j0+0)*nx],
                    normalized[(x+5) + (j0+0)*nx],
                    normalized[(x+6) + (j0+0)*nx],
                    normalized[(x+7) + (j0+0)*nx]
                };

                float8_t b1 = {
                    normalized[(x+0) + (j0+1)*nx],
                    normalized[(x+1) + (j0+1)*nx],
                    normalized[(x+2) + (j0+1)*nx],
                    normalized[(x+3) + (j0+1)*nx],
                    normalized[(x+4) + (j0+1)*nx],
                    normalized[(x+5) + (j0+1)*nx],
                    normalized[(x+6) + (j0+1)*nx],
                    normalized[(x+7) + (j0+1)*nx]
                };

                float8_t b2 = {
                    normalized[(x+0) + (j0+2)*nx],
                    normalized[(x+1) + (j0+2)*nx],
                    normalized[(x+2) + (j0+2)*nx],
                    normalized[(x+3) + (j0+2)*nx],
                    normalized[(x+4) + (j0+2)*nx],
                    normalized[(x+5) + (j0+2)*nx],
                    normalized[(x+6) + (j0+2)*nx],
                    normalized[(x+7) + (j0+2)*nx]
                };

                float8_t b3 = {
                    normalized[(x+0) + (j0+3)*nx],
                    normalized[(x+1) + (j0+3)*nx],
                    normalized[(x+2) + (j0+3)*nx],
                    normalized[(x+3) + (j0+3)*nx],
                    normalized[(x+4) + (j0+3)*nx],
                    normalized[(x+5) + (j0+3)*nx],
                    normalized[(x+6) + (j0+3)*nx],
                    normalized[(x+7) + (j0+3)*nx]
                };
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
            
            for (; x < nx; x++) {
                float a0 = normalized[x + (i0+0)*nx];
                float a1 = normalized[x + (i0+1)*nx];
                float a2 = normalized[x + (i0+2)*nx];
                float a3 = normalized[x + (i0+3)*nx];

                float b0 = normalized[x + (j0+0)*nx];
                float b1 = normalized[x + (j0+1)*nx];
                float b2 = normalized[x + (j0+2)*nx];
                float b3 = normalized[x + (j0+3)*nx];
                
                rest00 += a0 * b0;
                rest01 += a0 * b1;
                rest02 += a0 * b2;
                rest03 += a0 * b3;

                rest10 += a1 * b0;
                rest11 += a1 * b1;
                rest12 += a1 * b2;
                rest13 += a1 * b3;

                rest20 += a2 * b0;
                rest21 += a2 * b1;
                rest22 += a2 * b2;
                rest23 += a2 * b3;

                rest30 += a3 * b0;
                rest31 += a3 * b1;
                rest32 += a3 * b2;
                rest33 += a3 * b3;
            }

            float dot00 = sum00[0] + sum00[1] + sum00[2] + sum00[3] + sum00[4] + sum00[5] + sum00[6] + sum00[7] + rest00;
            float dot01 = sum01[0] + sum01[1] + sum01[2] + sum01[3] + sum01[4] + sum01[5] + sum01[6] + sum01[7] + rest01;
            float dot02 = sum02[0] + sum02[1] + sum02[2] + sum02[3] + sum02[4] + sum02[5] + sum02[6] + sum02[7] + rest02;
            float dot03 = sum03[0] + sum03[1] + sum03[2] + sum03[3] + sum03[4] + sum03[5] + sum03[6] + sum03[7] + rest03;

            float dot10 = sum10[0] + sum10[1] + sum10[2] + sum10[3] + sum10[4] + sum10[5] + sum10[6] + sum10[7] + rest10;
            float dot11 = sum11[0] + sum11[1] + sum11[2] + sum11[3] + sum11[4] + sum11[5] + sum11[6] + sum11[7] + rest11;
            float dot12 = sum12[0] + sum12[1] + sum12[2] + sum12[3] + sum12[4] + sum12[5] + sum12[6] + sum12[7] + rest12;
            float dot13 = sum13[0] + sum13[1] + sum13[2] + sum13[3] + sum13[4] + sum13[5] + sum13[6] + sum13[7] + rest13;

            float dot20 = sum20[0] + sum20[1] + sum20[2] + sum20[3] + sum20[4] + sum20[5] + sum20[6] + sum20[7] + rest20;
            float dot21 = sum21[0] + sum21[1] + sum21[2] + sum21[3] + sum21[4] + sum21[5] + sum21[6] + sum21[7] + rest21;
            float dot22 = sum22[0] + sum22[1] + sum22[2] + sum22[3] + sum22[4] + sum22[5] + sum22[6] + sum22[7] + rest22;
            float dot23 = sum23[0] + sum23[1] + sum23[2] + sum23[3] + sum23[4] + sum23[5] + sum23[6] + sum23[7] + rest23;

            float dot30 = sum30[0] + sum30[1] + sum30[2] + sum30[3] + sum30[4] + sum30[5] + sum30[6] + sum30[7] + rest30;
            float dot31 = sum31[0] + sum31[1] + sum31[2] + sum31[3] + sum31[4] + sum31[5] + sum31[6] + sum31[7] + rest31;
            float dot32 = sum32[0] + sum32[1] + sum32[2] + sum32[3] + sum32[4] + sum32[5] + sum32[6] + sum32[7] + rest32;
            float dot33 = sum33[0] + sum33[1] + sum33[2] + sum33[3] + sum33[4] + sum33[5] + sum33[6] + sum33[7] + rest33;

            result[(i0+0) + (j0+0)*ny] = dot00;
            result[(i0+0) + (j0+1)*ny] = dot01;
            result[(i0+0) + (j0+2)*ny] = dot02;
            result[(i0+0) + (j0+3)*ny] = dot03;

            result[(i0+1) + (j0+0)*ny] = dot10;
            result[(i0+1) + (j0+1)*ny] = dot11;
            result[(i0+1) + (j0+2)*ny] = dot12;
            result[(i0+1) + (j0+3)*ny] = dot13;

            result[(i0+2) + (j0+0)*ny] = dot20;
            result[(i0+2) + (j0+1)*ny] = dot21;
            result[(i0+2) + (j0+2)*ny] = dot22;
            result[(i0+2) + (j0+3)*ny] = dot23;

            result[(i0+3) + (j0+0)*ny] = dot30;
            result[(i0+3) + (j0+1)*ny] = dot31;
            result[(i0+3) + (j0+2)*ny] = dot32;
            result[(i0+3) + (j0+3)*ny] = dot33;
        }

        // #pragma omp parallel for schedule(dynamic, 1)
        for (int ii = i0; ii <= i0 + 3; ii++) {
            for (int j = i0; j <= ii; j++) {
                float8_t dot4 = {0, 0, 0, 0, 0, 0, 0, 0};
                float rest = 0;

                int x = 0;

                // for (; x + 3 < nx; x += 4) {
                for (; x + 7 < nx; x += 8) {
                    float8_t a = {
                        normalized[x + ii*nx],
                        normalized[(x + 1) + ii*nx],
                        normalized[(x + 2) + ii*nx],
                        normalized[(x + 3) + ii*nx],
                        normalized[(x + 4) + ii*nx],
                        normalized[(x + 5) + ii*nx],
                        normalized[(x + 6) + ii*nx],
                        normalized[(x + 7) + ii*nx]
                    };

                    float8_t b = {
                        normalized[x + j*nx],
                        normalized[(x + 1) + j*nx],
                        normalized[(x + 2) + j*nx],
                        normalized[(x + 3) + j*nx],
                        normalized[(x + 4) + j*nx],
                        normalized[(x + 5) + j*nx],
                        normalized[(x + 6) + j*nx],
                        normalized[(x + 7) + j*nx]
                    };

                    dot4 += a * b;
                }

                for (; x < nx; x++) {
                    rest += normalized[x + ii*nx] * normalized[x + j*nx];
                }

                float dot = dot4[0] + dot4[1] + dot4[2] + dot4[3] + dot4[4] + dot4[5] + dot4[6] + dot4[7] + rest;

                result[ii + j*ny] = dot;
            }
        }
    }
    #pragma omp parallel for schedule(dynamic, 1)
    for (int ii = ny - (ny % 4); ii < ny; ii++) {
        for (int j = 0; j <= ii; j++) {
            float8_t dot4 = {0, 0, 0, 0, 0, 0, 0, 0};
            float rest = 0;
            
            int x = 0;

            for (; x + 7 < nx; x+= 8) {
                float8_t a = {
                    normalized[x + ii*nx],
                    normalized[(x + 1) + ii*nx],
                    normalized[(x + 2) + ii*nx],
                    normalized[(x + 3) + ii*nx],
                    normalized[(x + 4) + ii*nx],
                    normalized[(x + 5) + ii*nx],
                    normalized[(x + 6) + ii*nx],
                    normalized[(x + 7) + ii*nx]
                };

                float8_t b = {
                    normalized[x + j*nx],
                    normalized[(x + 1) + j*nx],
                    normalized[(x + 2) + j*nx],
                    normalized[(x + 3) + j*nx],
                    normalized[(x + 4) + j*nx],
                    normalized[(x + 5) + j*nx],
                    normalized[(x + 6) + j*nx],
                    normalized[(x + 7) + j*nx]
                };
                dot4 += a * b;
            }

            for (; x < nx; x++) {
                rest += normalized[x + ii*nx] * normalized[x + j*nx];
            }

            float dot = dot4[0] + dot4[1] + dot4[2] + dot4[3] + dot4[4] + dot4[5] + dot4[6] + dot4[7] + rest;

            result[ii + j*ny] = dot;
        }
    }
    delete[] normalized;
}
