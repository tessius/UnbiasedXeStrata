# Lifted from XeStrata (https://github.com/MistVVK/XeStrata, commit 721afe5)

`iq_mmq.cpp` (+ matrix_report.cpp and the small headers): XeStrata's SYCL port of llama.cpp's quantized matrix tiles (dp4a and
Intel XMX), used here as strata-poly's `Mmq` (../poly/mmq_sycl.cpp). Only the include paths changed (flattened into this
directory; `xe_types.hpp` replaces the two constants taken from XeStrata's native_mmvq.hpp).

License: these files are XeStrata's: SPDX `LGPL-3.0-or-later AND MIT` (see their headers; the ggml parts are MIT, (c) The ggml
authors). The rest of this repository is MIT. A binary that links them is covered by the LGPL for these files: keep this
directory separable (it is a static library, `sp_mmq`).

## What was changed in the lifted files (LGPL-3.0 section 4: modified files are marked)
Taken from XeStrata commit 721afe51f223fe5e60470a06a22c9961c7625fa8 (2026-10-30, "upstream Strata 0.2.8 integration recorded").
Unmodified except for include paths: amd_wmma.hpp, cuda_intrinsics.hpp, device_target.hpp, iq_bits.hpp, iq_mmq.hpp,
matrix_report.cpp, matrix_report.hpp, per_device.hpp, device_arch_tag.hpp (`#include "strata/..."` became `#include "..."`).
`iq_mmq.cpp`: include paths as above, `native_mmvq.hpp` replaced by `xe_types.hpp` (two constants), and these edits, all under
`#ifdef STRATA_MMQ_NO_XE_HPG_SHAPE` (set by sycl-poly's CMake for an ahead-of-time build for Xe2 / Battlemage, because that
backend cannot compile the Xe-HPG and the generic matrix shapes): the `is == 2` Intel shape is not built, the generic matrix
fallback is replaced by dp4a, `intel_fits_is` / `intel_layout_ok_is` / `kMaxIntelShape` select shape 1 only, and `iq_mmq_usable`
returns the Intel answer directly.  Without that macro the file behaves as XeStrata's.
New (ours): `xe_types.hpp`, `../poly/mmq_sycl.cpp` (poly's `Mmq` class over `iq_mmq_quantize` / `iq_mmq_grouped`).
`COPYING` / `COPYING.LESSER` / `NOTICE.XeStrata` are XeStrata's, unchanged: its additional permission (linking with vendor math
libraries such as oneMKL) is in NOTICE.XeStrata.
