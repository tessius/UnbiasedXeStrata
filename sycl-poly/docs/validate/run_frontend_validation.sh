#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
# SPDX-License-Identifier: MIT
# Starts unbiased-strata with the SYCL engine, runs frontend_checks.py against it, stops it.
#   FRONTEND=build-frontend/bin/unbiased-strata ENGINE=build/bin/strata-poly MODEL=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
#   [MMPROJ=mmproj-F16.gguf] docs/validate/run_frontend_validation.sh
set -u
: "${FRONTEND:?}" "${ENGINE:?}" "${MODEL:?}"
here=$(cd "$(dirname "$0")" && pwd); port=${PORT:-8090}
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}
log=$(mktemp); echo "== start: $FRONTEND --engine $(basename "$ENGINE") -m $(basename "$MODEL")${MMPROJ:+ --mmproj $(basename "$MMPROJ")}"
"$FRONTEND" --engine "$ENGINE" -m "$MODEL" ${MMPROJ:+--mmproj "$MMPROJ" --threads 8} --port "$port" --host 127.0.0.1 -- --model "$MODEL" --max-context 8192 --no-graphs > "$log" 2>&1 &
pid=$!; trap 'kill -TERM $pid 2>/dev/null; wait $pid 2>/dev/null; rm -f "$log"' EXIT
for i in $(seq 1 150); do curl -s -m 3 "http://127.0.0.1:$port/health" | grep -q '"status":"ok"' && break; kill -0 $pid 2>/dev/null || { echo "server died:"; tail -n 5 "$log"; exit 2; }; sleep 4; done
curl -s "http://127.0.0.1:$port/health"; echo
python3 "$here/frontend_checks.py" "http://127.0.0.1:$port" ${MMPROJ:+} $([ -z "${MMPROJ:-}" ] && echo --no-vision)
