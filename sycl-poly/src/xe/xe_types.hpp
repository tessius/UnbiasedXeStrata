// The engine-private type ids iq_mmq.cpp names (from XeStrata's include/strata/kernels/native_mmvq.hpp). In strata-poly the
// weights stay in GGUF order, so these row-split layouts are never produced: only the constants are needed to compile.
#pragma once
namespace strata::kernels {
constexpr int kXeTypeBase = 0x58530000;
constexpr int kNativeQ6KRows = kXeTypeBase + 1;
constexpr int kNativeQ8Rows = kXeTypeBase + 2;
}  // namespace strata::kernels
