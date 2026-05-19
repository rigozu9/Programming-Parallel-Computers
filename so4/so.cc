#include <algorithm>
#include <omp.h>

typedef unsigned long long data_t;

void psort(int n, data_t *data) {
    if (n <= 1) {
        return;
    }
    // FIXME: Implement a more efficient parallel sorting algorithm for the CPU,
    // using the basic idea of merge sort.
    // std::sort(data, data + n);
    int threads = omp_get_max_threads();
    int chunk_size = (n + threads - 1) / threads;

    #pragma omp parallel for
    for (int t = 0; t < threads; t++) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, n);
        if (start < n) {
            std::sort(data + start, data + end);
        }
    }

    data_t* tmp = new data_t[n];

    data_t* src = data;
    data_t* dst = tmp;

    for (; chunk_size < n; chunk_size *= 2) {
        int step = 2 * chunk_size;
        int blocks = (n + step - 1) / step;

        #pragma omp parallel for
        for (int b = 0; b < blocks; b++) {
            int left = b * step;
            int mid = std::min(left + chunk_size, n);
            int right = std::min(left + step, n);

            std::merge(src + left, src + mid, src + mid, src + right, dst + left);
        }
        std::swap(src, dst);
    }
    if (src != data) {
        std::copy(src, src + n, data);
    }
    delete[] tmp;
}
