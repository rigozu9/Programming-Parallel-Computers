#include <vector>
#include <cfloat>

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

    double best_cost = DBL_MAX;

    int width = nx + 1;
    int height = ny + 1;

    std::vector<double> sum_of_color(3 * width * height, 0.0);
    std::vector<double> sum_of_squares(3 * width * height, 0.0);

    auto idx = [&](int y, int x, int c) {
        return c + 3 * x + 3 * width * y;
    };

    auto rectangle_sum = [&](int y0, int x0, int y1, int x1, int c) {
        return sum_of_color[idx(y1, x1, c)] - sum_of_color[idx(y0, x1, c)] - sum_of_color[idx(y1, x0, c)] + sum_of_color[idx(y0, x0, c)];
    };

    auto rectangle_sum_of_squares = [&](int y0, int x0, int y1, int x1, int c) {
        return sum_of_squares[idx(y1, x1, c)] - sum_of_squares[idx(y0, x1, c)] - sum_of_squares[idx(y1, x0, c)] + sum_of_squares[idx(y0, x0, c)];
    };

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            for (int c = 0; c < 3; ++c) {
                double value = data[c + 3 * x + 3 * nx * y];
                double squared_value = value * value;
                
                sum_of_color[idx(y + 1, x + 1, c)] = value + sum_of_color[idx(y,x + 1, c)] + sum_of_color[idx(y + 1, x,c)] - sum_of_color[idx(y,x,c)];
                sum_of_squares[idx(y + 1, x + 1, c)] = squared_value + sum_of_squares[idx(y,x + 1, c)] + sum_of_squares[idx(y + 1, x,c)] - sum_of_squares[idx(y,x,c)];
            }
        }
    }
    for (int y0 = 0; y0 < ny; ++y0) {
        for (int y1 = y0 + 1; y1 <= ny; ++y1) {
            for (int x0 = 0; x0 < nx; ++x0) {
                for (int x1 = x0 + 1; x1 <= nx; ++x1) {
                    int inner_count = (y1 - y0) * (x1 - x0);
                    int total_count = ny * nx;
                    int outer_count = total_count - inner_count;

                    double cost = 0.0;

                    for (int c = 0; c < 3; ++c) {
                        double whole_image_sum = rectangle_sum(0, 0, ny, nx, c);
                        double inner_sum = rectangle_sum(y0, x0, y1, x1, c);
                        double outer_sum = whole_image_sum - inner_sum;

                        double whole_image_sum_of_squares = rectangle_sum_of_squares(0, 0, ny, nx, c);
                        double inner_sum_of_squares = rectangle_sum_of_squares(y0, x0, y1, x1, c);
                        double outer_sum_of_squares = whole_image_sum_of_squares - inner_sum_of_squares;

                        double inner_error = inner_sum_of_squares - inner_sum * inner_sum / inner_count;
                        
                        double outer_error = 0.0;

                        if (outer_count > 0) {
                            outer_error =
                                outer_sum_of_squares - outer_sum * outer_sum / outer_count;
                        }

                        cost += inner_error + outer_error;
                    }
                    if (cost < best_cost) {
                        best_cost = cost;

                        result.y0 = y0;
                        result.x0 = x0;
                        result.y1 = y1;
                        result.x1 = x1;
                        for (int c = 0; c < 3; ++c) {
                            double whole_image_sum = rectangle_sum(0, 0, ny, nx, c);
                            double inner_sum = rectangle_sum(y0, x0, y1, x1, c);
                            double outer_sum = whole_image_sum - inner_sum;

                            result.inner[c] = inner_sum / inner_count;

                            if (outer_count > 0) {
                                result.outer[c] = outer_sum / outer_count;
                            } else {
                                result.outer[c] = 0.0f;
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}