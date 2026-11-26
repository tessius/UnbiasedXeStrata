#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# The machine code of a kernel and the instruction mix of its K loop (what a timing cannot say: DEVLOG sections 33-40).
#   sycl-poly/tools/isa_mix.sh [path/to/iq_mmq.cpp] [mangled kernel name]
# Builds tools/isa_drv.cpp with the given iq_mmq.cpp (default: the tree's) for the Arc Pro B60 (bmg-g21), keeps the device binaries (-save-temps), disassembles them
# with ocloc, and prints the instruction mix of the loop of the XMX int8 Q4_K tile, Kernel<12, 2, 2, false, false, 1>.  The disassembly stays in $OUT (default ./isa_out).
set -eu
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd)
src=${1:-$root/sycl-poly/src/xe/iq_mmq.cpp}; name=${2:-_ZTSN6strata7kernels12_GLOBAL__N_16KernelILi12ELi2ELi2ELb0ELb0ELi1EEE}; out=${OUT:-isa_out}
mkdir -p "$out"; cd "$out"
icpx -O3 -DNDEBUG -std=gnu++20 -DSTRATA_MMQ_NO_XE_HPG_SHAPE=1 -DSTRATA_VERSION=\"x\" -I"$root/sycl-poly/src/xe" -I"$root/third_party/llama.cpp/ggml/src" -I"$root/third_party/llama.cpp/ggml/include" \
    -fsycl -fsycl-default-sub-group-size=32 -fp-model=precise -w -fsycl-targets=spir64_gen -Xspirv-translator --spirv-ext=+SPV_KHR_integer_dot_product \
    -Xsycl-target-backend=spir64_gen "-device bmg-g21 -options -cl-fp32-correctly-rounded-divide-sqrt" -save-temps "$here/isa_drv.cpp" "$src" "$root/sycl-poly/src/xe/matrix_report.cpp" -o drv
for f in *.out; do ocloc disasm -file "$f" -device bmg-g21 -dump "dis_$f" > /dev/null 2>&1 || true; done
F=$(ls dis_*/".text.$name.asm" | head -1)
python3 - "$F" <<'PY'
import re, sys, collections
lines = open(sys.argv[1], errors='replace').read().split('\n')
label = {m.group(1): n for n, l in enumerate(lines) for m in [re.match(r'^(L\d+):', l)] if m}
back = [(label[m.group(1)], n) for n, l in enumerate(lines) for m in [re.search(r'jmpi\s+(L\d+)', l)] if m and m.group(1) in label and label[m.group(1)] < n]
for a, b in back:
    c = collections.Counter(); sends = collections.Counter()
    for l in lines[a:b + 1]:
        m = re.match(r'^\s*(\([^)]*\))?\s*([a-z_0-9.]+)\s', l)
        if not m or l.startswith('L'): continue
        op = m.group(2); k = 'send' if op.startswith('send') else op.split('.')[0]; c[k] += 1
        if op.startswith('send'):
            mm = re.search(r'// .*?; (\S+)', l); sends[mm.group(1) if mm else '?'] += 1
    print(f'loop: {sum(c.values())} instructions:', dict(c.most_common(10))); print('  sends:', dict(sends.most_common(8)))
PY
