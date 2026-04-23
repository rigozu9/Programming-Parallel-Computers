#include <vector>
#include <algorithm>
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/
void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {
  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {

      std::vector<float> vals;

      int left = std::max(0, x - hx);
      int right = std::min(nx, x + hx + 1);
      int top = std::max(0, y - hy);
      int bottom = std::min(ny, y + hy + 1);

      for (int j = top; j < bottom; j++) {
        for (int i = left; i < right; i++) {
          vals.push_back(in[i + j * nx]);
        }
      }
      int n = vals.size();
      double median = 0.0;
      if (n % 2 == 1) {
        std::nth_element(vals.begin(), vals.begin() + n/2, vals.end());
        median = vals[n/2];
      } else {
        std::nth_element(vals.begin(), vals.begin() + n/2, vals.end());
        double value1 = vals[n/2];
        std::nth_element(vals.begin(), vals.begin() + n/2 - 1, vals.end());
        double value2 = vals[n/2 - 1];
        median = (value1 + value2) / 2;
      }
      out[x + nx*y] = median;
    }
  }
}