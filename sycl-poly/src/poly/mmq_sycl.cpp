// Mmq (src/poly/mmq.hpp) without llama.cpp's ggml-cuda tiles: fits() is false for every type, so the engine's prompt
// path takes its other products. TODO: a SYCL product here (XeStrata's iq_mmq / dp4a_gemm / xmx_gemm are the model).
#include "mmq.hpp"

#include <stdexcept>

namespace strata::poly {

Mmq::Mmq() {}
Mmq::~Mmq() {}
bool Mmq::fits(int, int64_t, int64_t) { return false; }
size_t Mmq::q8_bytes(int64_t rows, int64_t cols) { return (size_t) rows * (size_t) ((cols + 31) / 32) * 36; }
void Mmq::quantize(const float*, const int32_t*, void*, int, int64_t, int64_t, int64_t, void*) {
    throw std::runtime_error("Mmq: not built for SYCL");
}
void Mmq::mul(const void*, int, int64_t, int64_t, size_t, int, const void*, const int32_t*, const int32_t*, int64_t,
              int64_t, float*, int64_t, void*) {
    throw std::runtime_error("Mmq: not built for SYCL");
}
bool Mmq::tab(int) { return false; }
bool Mmq::mul_tab(const void* const*, int, int64_t, int64_t, int, const void*, const int32_t*, const int32_t*, int64_t,
                  int64_t, float*, int64_t, void*) {
    return false;
}

}  // namespace strata::poly
