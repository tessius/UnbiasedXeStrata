#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
# SPDX-License-Identifier: MIT
# End-to-end validation of a strata-poly SYCL build.  Writes everything to stdout; tee it into docs/logs/.
#   BIN=build-poly/strata-poly MODEL=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf TOKENIZE=/path/to/llama-tokenize \
#   BUILD=build-poly  docs/validate/run_validation.sh
# BUILD (optional) is the build directory that has the poly_*_q_test unit-test binaries.
set -u
: "${BIN:?}" "${MODEL:?}" "${TOKENIZE:?}"
here=$(cd "$(dirname "$0")" && pwd); tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu} BIN MODEL TOKENIZE
fail=0
echo "== environment"; date -u +%FT%TZ; uname -srm; sycl-ls 2>/dev/null | grep -i level_zero | head -2
"$BIN" --help 2>&1 | head -1
echo "== kernel unit tests against CPU references"
if [ -n "${BUILD:-}" ]; then
  for t in poly_gdn_q_test poly_attend_q_test poly_rope_q_test; do
    echo "-- $t"; "$BUILD/$t" 2>&1 | tail -n 4; [ "${PIPESTATUS[0]}" = 0 ] || { echo "FAILED: $t"; fail=1; }
  done
fi
echo "== teacher-forced counting (positions 50..190)"
python3 "$here/teacher_force.py" || fail=1
echo "== numbered-facts retrieval (answer must be the exact fact)"
for n in 12 100; do
  python3 "$here/mkfacts.py" $n > "$tmp/f$n.txt" 2> "$tmp/f$n.expect"
  ids=$("$TOKENIZE" -m "$MODEL" --ids --no-bos -f "$tmp/f$n.txt" 2>/dev/null | tr -d '[] \n')
  want=$("$TOKENIZE" -m "$MODEL" --ids --no-bos -p "$(cat "$tmp/f$n.expect")" 2>/dev/null | tr -d '[] \n')
  out=$("$BIN" --model "$MODEL" --tokens "$ids" --max-new 80 --no-graphs --no-draft --max-context 4096 2>"$tmp/err$n")
  got=$(echo "$out" | tail -n 1)
  echo "facts=$n prompt_tokens=$(echo "$ids" | tr ',' '\n' | wc -l): $(grep -a '^prompt' "$tmp/err$n" | cut -c1-120)"
  case "$got" in "$want",248046) echo "  exact: ok";; *) echo "  WRONG: want $want got $got"; fail=1;; esac
done
echo "== counting 1..200"
ids=$("$TOKENIZE" -m "$MODEL" --ids --no-bos -p $'<|im_start|>user\nCount from 1 to 200, one number per line, nothing else.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' 2>/dev/null | tr -d '[] \n')
"$BIN" --model "$MODEL" --tokens "$ids" --max-new 800 --no-graphs --max-context 4096 > "$tmp/c.out" 2> "$tmp/c.err"
grep -a '^prompt' "$tmp/c.err" | cut -c1-120
python3 - "$tmp/c.out" <<'PY' || fail=1
import sys
ids=[int(x) for x in open(sys.argv[1]).read().strip().split('\n')[-1].split(',')]
s="";
for t in ids: s+="\n" if t==198 else (str(t-15) if 15<=t<=24 else "<%d>"%t)
want="".join(f"{i}\n" for i in range(1,200))+"200<248046>"
print("counting exact:", "ok" if s==want else "WRONG: "+repr(s[-60:])); sys.exit(0 if s==want else 1)
PY
echo "== speed (3 runs each; prompt = 100 numbered facts)"
python3 "$here/mkfacts.py" 100 > "$tmp/f.txt" 2>/dev/null; ids=$("$TOKENIZE" -m "$MODEL" --ids --no-bos -f "$tmp/f.txt" 2>/dev/null | tr -d '[] \n')
for extra in "--no-draft" ""; do
  for i in 1 2 3; do echo "[$extra] $("$BIN" --model "$MODEL" --tokens "$ids" --max-new 60 --no-graphs --max-context 4096 $extra 2>&1 | grep -a '^prompt' | cut -c1-110)"; done
done
echo "== result: $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
