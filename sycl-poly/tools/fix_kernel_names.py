#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
# SPDX-License-Identifier: MIT
"""dpct names the kernel of a templated `*_wrapper` function `dpct_kernel_name<class X_>` with no template arguments,
so every instantiation gets the same name (the SYCL integration header then fails: redefinition of KernelInfo).
Give the name the wrapper's own template parameters: types as they are, values as dpct_kernel_scalar<V>.
Idempotent.  usage: fix_kernel_names.py FILE..."""
import re, sys

TPL = re.compile(r"template\s*<([^>]*(?:<[^>]*>[^>]*)*)>\s*(?:/\*.*?\*/\s*)*(?:__dpct_inline__\s+)?void\s+(\w+_wrapper)\s*\(", re.S)
BARE = re.compile(r"dpct_kernel_name<class (\w+_)>>")

def params(lst):
    out = []
    for p in re.split(r",(?![^<]*>)", lst):
        p = p.split("=")[0].strip()
        if not p: continue
        m = re.match(r"(typename|class)\s+(\w+)", p)
        out.append((True, m.group(2)) if m else (False, p.split()[-1]))
    return out

for path in sys.argv[1:]:
    s = open(path).read()
    pos, n = 0, 0
    res = []
    for m in TPL.finditer(s):
        # the wrapper's body: from its "(" to the next top-level "\n}\n"
        end = s.find("\n}\n", m.end())
        body = s[m.end():end]
        ps = params(m.group(1))
        def sub(b):
            args = "".join(", " + (n if t else f"dpct_kernel_scalar<{n}>") for t, n in ps)
            return f"dpct_kernel_name<class {b.group(1)}{args}>>"
        new, c = BARE.subn(sub, body)
        if c:
            res.append((m.end(), end, new)); n += c
    for a, b, new in reversed(res):
        s = s[:a] + new + s[b:]
    open(path, "w").write(s)
    print(f"{path}: {n} kernel names qualified")
