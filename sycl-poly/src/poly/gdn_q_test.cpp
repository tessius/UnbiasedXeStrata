// gdn_step() (Qwen3.6's gated delta rule: 128-value heads, 32 value heads over 16 key heads) for 300 single-token steps,
// the state carried from step to step, against a CPU reference of the same arithmetic.
#include "strata/poly/kernels.hpp"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <vector>

using namespace strata::poly;

int main() {
    const int dim = 128, n_head = 32, k_heads = 16, steps = 300;
    const float eps = getenv("EPS") ? (float) atof(getenv("EPS")) : 1e-6f;
    const size_t row_n = (2 * (size_t) k_heads + n_head) * dim, d_inner = (size_t) n_head * dim;
    sycl::queue& q = dpct::get_in_order_queue();
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> dt(n_head), a(n_head), onorm(dim), s_ref((size_t) n_head * dim * dim, 0.f);
    for (auto& v : dt) v = 0.1f * nd(rng);
    for (auto& v : a) v = -(0.1f + 2.f * std::fabs(nd(rng)));
    for (auto& v : onorm) v = 1.f + 0.1f * nd(rng);
    auto* d_dt = sycl::malloc_device<float>(n_head, q);
    auto* d_a = sycl::malloc_device<float>(n_head, q);
    auto* d_on = sycl::malloc_device<float>(dim, q);
    auto* d_s = sycl::malloc_device<float>(s_ref.size(), q);
    auto* d_qkv = sycl::malloc_device<float>(row_n, q);
    auto* d_al = sycl::malloc_device<float>(n_head, q);
    auto* d_be = sycl::malloc_device<float>(n_head, q);
    auto* d_z = sycl::malloc_device<float>(d_inner, q);
    auto* d_out = sycl::malloc_device<float>(d_inner, q);
    q.memcpy(d_dt, dt.data(), n_head * 4); q.memcpy(d_a, a.data(), n_head * 4); q.memcpy(d_on, onorm.data(), dim * 4);
    q.memset(d_s, 0, s_ref.size() * 4).wait();
    int first_bad = -1; double worst = 0;
    size_t keys_g = 0; std::vector<float> qn_g, dec_g, sh_g; std::vector<std::vector<double>> o_g(n_head);
    std::vector<float> qkv(row_n), al(n_head), be(n_head), z(d_inner), out(d_inner), ref(d_inner);
    for (int step = 0; step < steps; ++step) {
        for (auto& v : qkv) v = nd(rng);
        for (auto& v : al) v = nd(rng);
        for (auto& v : be) v = nd(rng);
        for (auto& v : z) v = nd(rng);
        // CPU reference
        std::vector<float> qn(qkv.begin(), qkv.end());
        for (int hk = 0; hk < k_heads; ++hk) {
            float* qp = qn.data() + (size_t) hk * dim; float* kp = qn.data() + (size_t) dim * k_heads + (size_t) hk * dim;
            double qs = 0, ks = 0;
            for (int i = 0; i < dim; ++i) { qs += (double) qp[i] * qp[i]; ks += (double) kp[i] * kp[i]; }
            for (int i = 0; i < dim; ++i) { qp[i] = (float) (qp[i] / std::sqrt(qs + eps) * (1.0 / std::sqrt((double) dim))); kp[i] = (float) (kp[i] / std::sqrt(ks + eps)); }
        }
        const size_t keys = (size_t) dim * k_heads;
        keys_g = keys; qn_g = qn; dec_g.assign(n_head, 0.f); sh_g.assign(n_head, 0.f);
        for (int h = 0; h < n_head; ++h) {
            const int hk = h % k_heads;
            const double g = al[h] + dt[h];
            const double sp = g > 20.0 ? g : std::log(1.0 + std::exp(g));
            const double dc = std::exp(sp * a[h]), sh = 1.0 / (1.0 + std::exp(-(double) be[h]));
            dec_g[h] = (float) dc; sh_g[h] = (float) sh;
            const float* qv = qn.data() + (size_t) hk * dim; const float* kv = qn.data() + keys + (size_t) hk * dim;
            std::vector<double> o(dim);
            for (int j = 0; j < dim; ++j) {
                float* col = s_ref.data() + ((size_t) h * dim + j) * dim;
                const double v = qn[2 * keys + (size_t) h * dim + j];
                double sk = 0; for (int i = 0; i < dim; ++i) sk += (double) col[i] * dc * kv[i];
                const double d = (v - sk) * sh;
                double acc = 0;
                for (int i = 0; i < dim; ++i) { const double n = col[i] * dc + kv[i] * d; col[i] = (float) n; acc += n * qv[i]; }
                o[j] = acc;
            }
            o_g[h] = o;
            double os = 0; for (int j = 0; j < dim; ++j) os += o[j] * o[j];
            for (int j = 0; j < dim; ++j) {
                const double gate = z[(size_t) h * dim + j];
                ref[(size_t) h * dim + j] = (float) (o[j] / std::sqrt(os / dim + eps) * onorm[j] * (gate / (1.0 + std::exp(-gate))));
            }
        }
        q.memcpy(d_qkv, qkv.data(), row_n * 4); q.memcpy(d_al, al.data(), n_head * 4); q.memcpy(d_be, be.data(), n_head * 4);
        q.memcpy(d_z, z.data(), d_inner * 4); q.wait();
        gdn_step(d_qkv, d_al, d_be, d_dt, d_a, d_s, d_z, d_on, d_out, dim, n_head, k_heads, 1, eps, &q);
        q.memcpy(out.data(), d_out, d_inner * 4).wait();
        if (step == 0) {
            for (int h : {0, 31}) {
                const double os_r = [&] { double t = 0; for (double v : o_g[h]) t += v * v; return t; }();
                const double r = out[(size_t) h * dim + 5] / ref[(size_t) h * dim + 5], os_k = os_r / (r * r);
                double c32 = 0, c64 = 0, c96 = 0; for (int j = 0; j < 96; ++j) { const double t = o_g[h][j] * o_g[h][j]; if (j < 32) c32 += t; if (j < 64) c64 += t; c96 += t; }
                std::printf("  head %2d: sum o^2 ref %.4f, implied by kernel %.4f | first 32 %.4f, 64 %.4f, 96 %.4f of 128 (per-warp: ", h, os_r, os_k, c32, c64, c96);
                for (int w = 0; w < 4; ++w) { double t = 0; for (int j = 32 * w; j < 32 * w + 32; ++j) t += o_g[h][j] * o_g[h][j]; std::printf("%.4f ", t); }
                std::printf(")\n");
                double rmin = 1e30, rmax = -1e30; int nbad = 0;
                for (int j = 0; j < dim; ++j) { const double r = out[(size_t) h * dim + j] / ref[(size_t) h * dim + j]; if (std::fabs(ref[(size_t) h * dim + j]) > 1e-3) { rmin = std::min(rmin, r); rmax = std::max(rmax, r); } nbad += std::fabs(out[(size_t) h * dim + j] - ref[(size_t) h * dim + j]) > 5e-3; }
                std::printf("  head %2d: out/ref ratio over columns: min %.4f max %.4f; %d of %d columns off\n", h, rmin, rmax, nbad, dim);
            }
            for (int h : {0, 1, 17, 31}) {
                double e = 0; int worst_j = 0; for (int j = 0; j < dim; ++j) { double d = std::fabs(out[(size_t) h * dim + j] - ref[(size_t) h * dim + j]); if (d > e) { e = d; worst_j = j; } }
                std::printf("  head %2d: max err %.3e at col %3d: out %.5f ref %.5f | col0 out %.5f ref %.5f | col1 out %.5f ref %.5f\n", h, e, worst_j,
                            out[(size_t) h * dim + worst_j], ref[(size_t) h * dim + worst_j], out[(size_t) h * dim], ref[(size_t) h * dim], out[(size_t) h * dim + 1], ref[(size_t) h * dim + 1]);
            }
            std::vector<float> gq(row_n), ga(n_head), gb(n_head);
            q.memcpy(gq.data(), d_qkv, row_n * 4); q.memcpy(ga.data(), d_al, n_head * 4); q.memcpy(gb.data(), d_be, n_head * 4); q.wait();
            double eq = 0, ev = 0, ed = 0, es = 0;
            for (size_t i = 0; i < 2 * keys_g; ++i) eq = std::max(eq, (double) std::fabs(gq[i] - qn_g[i]));
            for (size_t i = 2 * keys_g; i < row_n; ++i) ev = std::max(ev, (double) std::fabs(gq[i] - qn_g[i]));
            for (int h = 0; h < n_head; ++h) { ed = std::max(ed, (double) std::fabs(ga[h] - dec_g[h])); es = std::max(es, (double) std::fabs(gb[h] - sh_g[h])); }
            std::printf("  intermediates: q,k err %.2e  v err %.2e  decay err %.2e  share err %.2e\n", eq, ev, ed, es);
        }
        double err = 0; for (size_t i = 0; i < d_inner; ++i) err = std::max(err, (double) std::fabs(out[i] - ref[i]));
        worst = std::max(worst, err);
        if (err > 5e-3 && first_bad < 0) first_bad = step;
        if (step % 25 == 0 || step == first_bad) std::printf("step %3d max|err|=%.3e%s\n", step, err, err > 5e-3 ? "  <-- WRONG" : "");
    }
    if (first_bad < 0) std::printf("ok (worst %.2e)\n", worst);
    else std::printf("FAIL first wrong step %d\n", first_bad);
    return first_bad >= 0;
}
