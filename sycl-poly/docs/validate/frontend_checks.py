#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
# SPDX-License-Identifier: MIT
"""Checks a running `unbiased-strata` server the way a user or a coding agent would: chat, streaming, exact retrieval at
2.7k tokens, thinking mode, tool calls, and (with an mmproj) an image.  usage: frontend_checks.py [URL] [--no-vision]
Exit 0 when every check passes."""
import base64, json, struct, subprocess, sys, tempfile, time, urllib.request, zlib, os

URL = next((a for a in sys.argv[1:] if a.startswith("http")), "http://127.0.0.1:8090")
VISION = "--no-vision" not in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
fails = []

def post(body, timeout=900):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.time(); d = json.load(urllib.request.urlopen(req, timeout=timeout)); return d, time.time() - t

def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""), flush=True)
    if not ok: fails.append(name)

NOTHINK = {"enable_thinking": False}
d, w = post({"model": "x", "max_tokens": 60, "chat_template_kwargs": NOTHINK,
             "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]})
a = d["choices"][0]["message"]["content"]; tm = d["timings"]
check("chat", "Paris" in a, f"{a!r}, {tm['predicted_per_second']:.1f} tok/s")

req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps({"model": "x", "stream": True, "max_tokens": 40, "chat_template_kwargs": NOTHINK,
        "messages": [{"role": "user", "content": "Count from 1 to 10."}]}).encode(), {"content-type": "application/json"})
chunks = [l for l in urllib.request.urlopen(req, timeout=300).read().decode().split("\n") if l.startswith("data: ")]
text = "".join(json.loads(c[6:])["choices"][0]["delta"].get("content") or "" for c in chunks if c != "data: [DONE]" and json.loads(c[6:]).get("choices"))
check("streaming", len(chunks) > 5 and chunks[-1] == "data: [DONE]" and "10" in text, f"{len(chunks)} events, text {text!r}")

facts = subprocess.run([sys.executable, os.path.join(HERE, "mkfacts.py"), "100"], capture_output=True, text=True)
user = facts.stdout.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]; want = facts.stderr.strip()
d, w = post({"model": "x", "max_tokens": 80, "chat_template_kwargs": NOTHINK, "messages": [{"role": "user", "content": user}]})
a = d["choices"][0]["message"]["content"]; tm = d["timings"]
check("retrieval at 2.7k tokens is exact", want in a, f"prompt {tm['prompt_n']} tokens, prefill {tm['prompt_per_second']:.0f} tok/s, decode {tm['predicted_per_second']:.1f} tok/s")

d, w = post({"model": "x", "max_tokens": 600, "chat_template_kwargs": NOTHINK, "messages": [{"role": "user", "content": "Write a short story about a lighthouse keeper, about 300 words."}]})
tm = d["timings"]; check("long generation", tm["predicted_n"] > 200, f"{tm['predicted_n']} tokens, decode {tm['predicted_per_second']:.1f} tok/s")

d, w = post({"model": "x", "max_tokens": 700, "messages": [{"role": "user", "content": "What is 17 * 23? Think, then answer in one line."}]})
m = d["choices"][0]["message"]; check("thinking mode splits reasoning from the answer", bool(m.get("reasoning_content")) and "391" in ((m.get("content") or "") + m.get("reasoning_content", "")),
                                       f"{len(m.get('reasoning_content') or '')} chars of reasoning, answer {(m.get('content') or '')[:40]!r}")

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
d, w = post({"model": "x", "max_tokens": 200, "tools": tools, "chat_template_kwargs": NOTHINK, "messages": [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]})
c = d["choices"][0]; tc = (c["message"].get("tool_calls") or [{}])[0].get("function", {})
check("tool call", c["finish_reason"] == "tool_calls" and tc.get("name") == "get_weather" and json.loads(tc.get("arguments", "{}")).get("city") == "Paris", json.dumps(tc))

if VISION:
    W = H = 336; rows = []
    for y in range(H):
        r = bytearray([0])
        for x in range(W): r += bytes((255, 255, 255) if 150 <= y < 186 else (0, 170, 0) if x < W // 2 else (220, 0, 0))
        rows.append(bytes(r))
    ch = lambda t, b: struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(b"".join(rows))) + ch(b"IEND", b"")
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    d, w = post({"model": "x", "max_tokens": 120, "chat_template_kwargs": NOTHINK, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "Which colors are on the left and the right of this image, and is there a bar?"}]}]})
    a = d["choices"][0]["message"]["content"].lower()
    check("vision (green left, red right, white bar)", "green" in a and "red" in a and "white" in a, a[:100])

print("front end checks:", "PASS" if not fails else "FAIL: " + ", ".join(fails))
sys.exit(1 if fails else 0)
