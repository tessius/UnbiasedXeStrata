# Lifted from XeStrata (https://github.com/MistVVK/XeStrata, commit 721afe5)

`iq_mmq.cpp` (+ matrix_report.cpp and the small headers): XeStrata's SYCL port of llama.cpp's quantized matrix tiles (dp4a and
Intel XMX), used here as strata-poly's `Mmq` (../poly/mmq_sycl.cpp). Only the include paths changed (flattened into this
directory; `xe_types.hpp` replaces the two constants taken from XeStrata's native_mmvq.hpp).

License: these files are XeStrata's: SPDX `LGPL-3.0-or-later AND MIT` (see their headers; the ggml parts are MIT, (c) The ggml
authors). The rest of this repository is MIT. A binary that links them is covered by the LGPL for these files: keep this
directory separable (it is a static library, `sp_mmq`).
