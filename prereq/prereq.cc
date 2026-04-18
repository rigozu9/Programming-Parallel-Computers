struct Result {
    float avg[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- horizontal position: 0 <= x0 < x1 <= nx
- vertical position: 0 <= y0 < y1 <= ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
- output: avg[c]
*/
Result calculate(int ny, int nx, const float *data, int y0, int x0, int y1, int x1) {
    Result result{{0.0f, 0.0f, 0.0f}};
    double sum_red = 0.0;
    double sum_green = 0.0;
    double sum_blue = 0.0;

    int width = x1 - x0;
    int height = y1 - y0;
    int count = width * height;

    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            int index = 3 * x + 3 * nx * y;
            sum_red += data[index];
            sum_green += data[index + 1];
            sum_blue += data[index + 2];
        }
    }
    result.avg[0] = (float)(sum_red / count);
    result.avg[1] = (float)(sum_green / count);
    result.avg[2] = (float)(sum_blue / count);
    return result;
}
