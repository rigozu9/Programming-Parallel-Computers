#include<cmath>
// Double rather than float because: Please do all arithmetic with double-precision floating point numbers.
typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    double* normalized = new double[ny * nx];

    #pragma omp parallel for
    for (int y = 0; y < ny; y++) {
        double sum = 0;

        for (int x = 0; x < nx; x++) {
            sum += data[x + y*nx];
        }

        double mean = sum / nx;

        double sq_sum = 0;

        for (int x = 0; x < nx; x++) {
            normalized[x + y*nx] = data[x + y*nx] - mean;
            sq_sum += normalized[x + y*nx] * normalized[x + y*nx];
        }

        double length = sqrt(sq_sum);

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
            double4_t sum00 = {0,0,0,0};
            double4_t sum01 = {0,0,0,0};
            double4_t sum02 = {0,0,0,0};
            double4_t sum03 = {0,0,0,0};

            double4_t sum10 = {0,0,0,0};
            double4_t sum11 = {0,0,0,0};
            double4_t sum12 = {0,0,0,0};
            double4_t sum13 = {0,0,0,0};

            double4_t sum20 = {0,0,0,0};
            double4_t sum21 = {0,0,0,0};
            double4_t sum22 = {0,0,0,0};
            double4_t sum23 = {0,0,0,0};

            double4_t sum30 = {0,0,0,0};
            double4_t sum31 = {0,0,0,0};
            double4_t sum32 = {0,0,0,0};
            double4_t sum33 = {0,0,0,0};

            double rest00 = 0;
            double rest01 = 0;
            double rest02 = 0; 
            double rest03 = 0;

            double rest10 = 0; 
            double rest11 = 0; 
            double rest12 = 0; 
            double rest13 = 0;

            double rest20 = 0; 
            double rest21 = 0;
            double rest22 = 0;
            double rest23 = 0;

            double rest30 = 0; 
            double rest31 = 0; 
            double rest32 = 0; 
            double rest33 = 0;
            
            int x = 0;

            for (; x + 3 < nx; x+= 4) {
                double4_t a0 = {
                    normalized[x + (i0+0)*nx],
                    normalized[(x+1) + (i0+0)*nx],
                    normalized[(x+2) + (i0+0)*nx],
                    normalized[(x+3) + (i0+0)*nx]
                };

                double4_t a1 = {
                    normalized[x + (i0+1)*nx],
                    normalized[(x+1) + (i0+1)*nx],
                    normalized[(x+2) + (i0+1)*nx],
                    normalized[(x+3) + (i0+1)*nx]
                };

                double4_t a2 = {
                    normalized[x + (i0+2)*nx],
                    normalized[(x+1) + (i0+2)*nx],
                    normalized[(x+2) + (i0+2)*nx],
                    normalized[(x+3) + (i0+2)*nx]
                };

                double4_t a3 = {
                    normalized[x + (i0+3)*nx],
                    normalized[(x+1) + (i0+3)*nx],
                    normalized[(x+2) + (i0+3)*nx],
                    normalized[(x+3) + (i0+3)*nx]
                };

                double4_t b0 = {
                    normalized[x + (j0+0)*nx],
                    normalized[(x+1) + (j0+0)*nx],
                    normalized[(x+2) + (j0+0)*nx],
                    normalized[(x+3) + (j0+0)*nx]
                };

                double4_t b1 = {
                    normalized[x + (j0+1)*nx],
                    normalized[(x+1) + (j0+1)*nx],
                    normalized[(x+2) + (j0+1)*nx],
                    normalized[(x+3) + (j0+1)*nx]
                };

                double4_t b2 = {
                    normalized[x + (j0+2)*nx],
                    normalized[(x+1) + (j0+2)*nx],
                    normalized[(x+2) + (j0+2)*nx],
                    normalized[(x+3) + (j0+2)*nx]
                };

                double4_t b3 = {
                    normalized[x + (j0+3)*nx],
                    normalized[(x+1) + (j0+3)*nx],
                    normalized[(x+2) + (j0+3)*nx],
                    normalized[(x+3) + (j0+3)*nx]
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
                double a0 = normalized[x + (i0+0)*nx];
                double a1 = normalized[x + (i0+1)*nx];
                double a2 = normalized[x + (i0+2)*nx];
                double a3 = normalized[x + (i0+3)*nx];

                double b0 = normalized[x + (j0+0)*nx];
                double b1 = normalized[x + (j0+1)*nx];
                double b2 = normalized[x + (j0+2)*nx];
                double b3 = normalized[x + (j0+3)*nx];
                
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

            double dot00 = sum00[0] + sum00[1] + sum00[2] + sum00[3] + rest00;
            double dot01 = sum01[0] + sum01[1] + sum01[2] + sum01[3] + rest01;
            double dot02 = sum02[0] + sum02[1] + sum02[2] + sum02[3] + rest02;
            double dot03 = sum03[0] + sum03[1] + sum03[2] + sum03[3] + rest03;

            double dot10 = sum10[0] + sum10[1] + sum10[2] + sum10[3] + rest10;
            double dot11 = sum11[0] + sum11[1] + sum11[2] + sum11[3] + rest11;
            double dot12 = sum12[0] + sum12[1] + sum12[2] + sum12[3] + rest12;
            double dot13 = sum13[0] + sum13[1] + sum13[2] + sum13[3] + rest13;

            double dot20 = sum20[0] + sum20[1] + sum20[2] + sum20[3] + rest20;
            double dot21 = sum21[0] + sum21[1] + sum21[2] + sum21[3] + rest21;
            double dot22 = sum22[0] + sum22[1] + sum22[2] + sum22[3] + rest22;
            double dot23 = sum23[0] + sum23[1] + sum23[2] + sum23[3] + rest23;

            double dot30 = sum30[0] + sum30[1] + sum30[2] + sum30[3] + rest30;
            double dot31 = sum31[0] + sum31[1] + sum31[2] + sum31[3] + rest31;
            double dot32 = sum32[0] + sum32[1] + sum32[2] + sum32[3] + rest32;
            double dot33 = sum33[0] + sum33[1] + sum33[2] + sum33[3] + rest33;

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
        #pragma omp parallel for schedule(dynamic, 1)
        for (int ii = i0; ii <= i0 + 3; ii++) {
            for (int j = i0; j <= ii; j++) {
                double4_t dot4 = {0, 0, 0, 0};
                double rest = 0;

                int x = 0;

                for (; x + 3 < nx; x += 4) {
                    double4_t a = {
                        normalized[x + ii*nx],
                        normalized[(x + 1) + ii*nx],
                        normalized[(x + 2) + ii*nx],
                        normalized[(x + 3) + ii*nx]
                    };

                    double4_t b = {
                        normalized[x + j*nx],
                        normalized[(x + 1) + j*nx],
                        normalized[(x + 2) + j*nx],
                        normalized[(x + 3) + j*nx]
                    };

                    dot4 += a * b;
                }

                for (; x < nx; x++) {
                    rest += normalized[x + ii*nx] * normalized[x + j*nx];
                }

                double dot = dot4[0] + dot4[1] + dot4[2] + dot4[3] + rest;

                result[ii + j*ny] = dot;
            }
        }
    }
    #pragma omp parallel for schedule(dynamic, 1)
    for (int ii = ny - (ny % 4); ii < ny; ii++) {
        for (int j = 0; j <= ii; j++) {
            double4_t dot4 = {0, 0, 0, 0};
            double dot0 = 0;
            
            int x = 0;

            for (; x + 3 < nx; x+= 4) {
                double4_t a = {
                    normalized[x + ii*nx],
                    normalized[(x + 1) + ii*nx],
                    normalized[(x + 2) + ii*nx],
                    normalized[(x + 3) + ii*nx]
                };

                double4_t b = {
                    normalized[x + j*nx],
                    normalized[(x + 1) + j*nx],
                    normalized[(x + 2) + j*nx],
                    normalized[(x + 3) + j*nx]
                };
                dot4 += a * b;
            }

            for (; x < nx; x++) {
                dot0 += normalized[x + ii*nx] * normalized[x + j*nx];
            }

            double dot = dot4[0] + dot4[1] + dot4[2] + dot4[3] + dot0;

            result[ii + j*ny] = dot;
        }
    }
    delete[] normalized;
}
