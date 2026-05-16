#include <cfloat>
typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};

    double best_score = -DBL_MAX;

    int width = nx + 1;
    int height = ny + 1;

    double4_t* sum_of_color = new double4_t[width * height];

    for (int i = 0; i < width * height; ++i) {
        sum_of_color[i] = double4_t{0.0, 0.0, 0.0, 0.0};
    }

    auto idx = [&](int y, int x) {
        return x + width * y;
    };

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            double4_t value{
                data[0 + 3 * x + 3 * nx * y],
                data[1 + 3 * x + 3 * nx * y],
                data[2 + 3 * x + 3 * nx * y],
                0.0
            };

            sum_of_color[idx(y + 1, x + 1)] = value + sum_of_color[idx(y,x + 1)] + sum_of_color[idx(y + 1,x)] - sum_of_color[idx(y,x)];
        }
    }

    double4_t total_sum = sum_of_color[idx(ny, nx)];
    int total_count = ny * nx;
    int band_count = ny * (ny + 1) / 2;

    #pragma omp parallel
    {
        double local_best_score = -DBL_MAX;
        Result local_result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};

        double4_t* strip = new double4_t[width];

        #pragma omp for schedule(dynamic, 1)
        for (int band = 0; band < band_count; ++band) {
            int rest = band;
            int y0 = 0;
            int choices = ny;

            while (rest >= choices) {
                rest -= choices;
                ++y0;
                --choices;
            }

            int y1 = y0 + 1 + rest;
            int h = y1 - y0;

            for (int x = 0; x <= nx; ++x) {
                strip[x] = sum_of_color[idx(y1, x)] - sum_of_color[idx(y0, x)];
            }

            for (int w = 1; w <= nx; ++w) {
                int inner_count = h * w;
                int outer_count = total_count - inner_count;

                double inv_inner = 1.0 / inner_count;
                double inv_outer = 0.0;

                if (outer_count > 0) {
                    inv_outer = 1.0 / outer_count;
                }

                for (int x0 = 0; x0 + w <= nx; ++x0) {
                    int x1 = x0 + w;

                    double4_t inner_sum = strip[x1] - strip[x0];
                    double4_t outer_sum = total_sum - inner_sum;

                    double score = 0.0;

                    score += (inner_sum[0] * inner_sum[0] + inner_sum[1] * inner_sum[1] + inner_sum[2] * inner_sum[2]) * inv_inner;

                    if (outer_count > 0) {
                        score += (outer_sum[0] * outer_sum[0] + outer_sum[1] * outer_sum[1] + outer_sum[2] * outer_sum[2]) * inv_outer;
                    }

                    if (score > local_best_score) {
                        local_best_score = score;

                        local_result.y0 = y0;
                        local_result.x0 = x0;
                        local_result.y1 = y1;
                        local_result.x1 = x1;

                        local_result.inner[0] = inner_sum[0] * inv_inner;
                        local_result.inner[1] = inner_sum[1] * inv_inner;
                        local_result.inner[2] = inner_sum[2] * inv_inner;

                        if (outer_count > 0) {
                            local_result.outer[0] = outer_sum[0] * inv_outer;
                            local_result.outer[1] = outer_sum[1] * inv_outer;
                            local_result.outer[2] = outer_sum[2] * inv_outer;
                        } else {
                            local_result.outer[0] = 0.0f;
                            local_result.outer[1] = 0.0f;
                            local_result.outer[2] = 0.0f;
                        }
                    }
                }
            }
        }

        delete[] strip;

        #pragma omp critical
        {
            if (local_best_score > best_score) {
                best_score = local_best_score;
                result = local_result;
            }
        }
    }

    delete[] sum_of_color;

    return result;
}