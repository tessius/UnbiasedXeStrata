#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
# SPDX-License-Identifier: MIT
"""Teacher-forced check: the model must predict the next token of a known sequence (1\\n2\\n3\\n ...) at many positions.
Needs: BIN (strata-poly), MODEL (gguf), TOKENIZE (llama.cpp's llama-tokenize).  Exit 0 when every position is right."""
import os, subprocess, sys
BIN, MODEL, TOK = os.environ["BIN"], os.environ["MODEL"], os.environ["TOKENIZE"]

def ids(text):
    out = subprocess.run([TOK, "-m", MODEL, "--ids", "--no-bos", "-p", text], capture_output=True, text=True).stdout
    return [int(x) for x in out.strip().strip("[]").replace(" ", "").split(",") if x]

prompt = "<|im_start|>user\nCount from 1 to 200, one number per line, nothing else.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ref = ids("".join(f"{i}\n" for i in range(1, 60)))
pi = ids(prompt)
good = n = 0
for k in (20, 33, 60, 63, 64, 65, 70, 100, 130, 160):
    toks = pi + ref[:k]
    r = subprocess.run([BIN, "--model", MODEL, "--tokens", ",".join(map(str, toks)), "--max-new", "1", "--no-graphs", "--no-draft",
                        "--max-context", "2048"], capture_output=True, text=True, timeout=600)
    got = r.stdout.strip().split("\n")[-1] if r.stdout.strip() else "FAIL"
    ok = got == str(ref[k]); good += ok; n += 1
    print(f"k={k:3d} position={len(toks):3d} want={ref[k]:4d} got={got:>5} {'ok' if ok else 'WRONG'}", flush=True)
print(f"teacher-forced: {good}/{n} correct")
sys.exit(0 if good == n else 1)
