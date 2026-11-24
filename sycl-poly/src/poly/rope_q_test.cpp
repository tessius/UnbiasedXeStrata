// head_norm_rope() for Qwen3.6 (head 256, 16 heads, 64 rotary values, theta 1e7) against a CPU reference at many positions.
#include "strata/poly/kernels.hpp"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace strata::poly;

int main() {
    const int dim = 256, heads = 16, n_rot = 64, half = n_rot / 2;
    const float eps = 1e-6f, theta_scale = std::pow(1e7f, -2.f / n_rot);
    sycl::queue& q = dpct::get_in_order_queue();
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) heads * dim), w(dim), y(x.size()), ref(x.size());
    for (auto& v : x) v = nd(rng);
    for (auto& v : w) v = 1.f + 0.1f * nd(rng);
    auto* dx = sycl::malloc_device<float>(x.size(), q);
    auto* dw = sycl::malloc_device<float>(w.size(), q);
    auto* dy = sycl::malloc_device<float>(y.size(), q);
    q.memcpy(dx, x.data(), x.size() * 4); q.memcpy(dw, w.data(), w.size() * 4); q.wait();
    int bad = 0, first_bad = -1;
    for (int pos = 0; pos < 700; ++pos) {
        head_norm_rope(dx, dim, dw, dy, dim, heads, n_rot, theta_scale, pos, 1, eps, &q);
        q.memcpy(y.data(), dy, y.size() * 4).wait();
        double err = 0;
        for (int h = 0; h < heads; ++h) {
            double ss = 0;
            for (int i = 0; i < dim; ++i) ss += (double) x[h * dim + i] * x[h * dim + i];
            const double scale = 1.0 / std::sqrt(ss / dim + eps);
            for (int i = 0; i < dim; ++i) {
                double r;
                if (i >= n_rot) r = x[h * dim + i] * scale * w[i];
                else {
                    const int j = i < half ? i : i - half;
                    const double theta = pos * std::pow((double) theta_scale, (double) j);
                    const double x0 = x[h * dim + j] * scale * w[j], x1 = x[h * dim + j + half] * scale * w[j + half];
                    r = i < half ? x0 * std::cos(theta) - x1 * std::sin(theta) : x0 * std::sin(theta) + x1 * std::cos(theta);
                }
                err = std::max(err, std::fabs(r - y[h * dim + i]));
            }
        }
        if (err > 2e-3) { ++bad; if (first_bad < 0) first_bad = pos; }
        if (pos % 50 == 0 || pos == 93 || pos == 94) std::printf("pos=%3d max|err|=%.3e%s\n", pos, err, err > 2e-3 ? "  <-- WRONG" : "");
    }
    std::printf("%d of 700 positions wrong%s\n", bad, bad ? "" : " - ok");
    if (bad) std::printf("first wrong position: %d\n", first_bad);
    return bad != 0;
}
