#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
# SPDX-License-Identifier: MIT
"""dpct names a kernel launched inside a TEMPLATE function `dpct_kernel_name<class X_>` with no template arguments, so every
instantiation of the function gets one name (the SYCL integration header then fails: redefinition of KernelInfo).  This gives the
name the enclosing template's own parameters: types as they are, values as dpct_kernel_scalar<V>.  Idempotent.
usage: fix_kernel_names_ik2.py FILE..."""
import re, sys

BARE = re.compile(r"dpct_kernel_name<\s*class (\w+_[0-9a-f]{6})\s*>>")

def params(lst):
    out = []
    depth = 0; cur = ""
    for ch in lst:
        if ch in "<(": depth += 1
        if ch in ">)": depth -= 1
        if ch == "," and depth == 0: out.append(cur); cur = ""
        else: cur += ch
    out.append(cur)
    res = []
    for p in out:
        p = p.split("=")[0].strip()
        if not p: continue
        m = re.match(r"(typename|class)\s*(\.\.\.)?\s*(\w+)", p)
        res.append(("type", m.group(3)) if m else ("val", p.split()[-1]))
    return res

def enclosing_template(s, pos):
    """The template parameter list of the function that contains `pos`, or None."""
    # the last function end at column 0 before pos bounds the search
    end = max(s.rfind("\n}\n", 0, pos), s.rfind("\n}\n\n", 0, pos), 0)
    best = None
    for m in re.finditer(r"^template\s*<", s[end:pos], re.M):
        best = end + m.start()
    if best is None: return None
    i = s.index("<", best); depth = 0
    for j in range(i, pos):
        if s[j] == "<": depth += 1
        elif s[j] == ">":
            depth -= 1
            if depth == 0: return params(s[i + 1:j])
    return None

for path in sys.argv[1:]:
    s = open(path).read(); out = []; last = 0; n = 0
    for m in BARE.finditer(s):
        ps = enclosing_template(s, m.start())
        if not ps: continue
        args = "".join(", " + (nm if k == "type" else f"dpct_kernel_scalar<{nm}>") for k, nm in ps)
        out.append(s[last:m.start()]); out.append(f"dpct_kernel_name<class {m.group(1)}{args}>>"); last = m.end(); n += 1
    out.append(s[last:]); open(path, "w").write("".join(out))
    print(f"{path}: {n} kernel names qualified")
