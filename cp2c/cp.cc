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
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {

            double4_t dot4 = {0, 0, 0, 0};

            double dot0 = 0;
            
            int x = 0;

            for (; x + 3 < nx; x+= 4) {
                double4_t a = {
                    normalized[x + i*nx],
                    normalized[(x + 1) + i*nx],
                    normalized[(x + 2) + i*nx],
                    normalized[(x + 3) + i*nx]
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
                dot0 += normalized[x + i*nx] * normalized[x + j*nx];
            }

            double dot = dot4[0] + dot4[1] + dot4[2] + dot4[3] + dot0;

            result[i + j*ny] = dot;
        }
    }
    delete[] normalized;
}
