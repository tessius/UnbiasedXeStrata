# SPDX-License-Identifier: MIT
# Greedy (temperature 0) answers to ten varied prompts from a running unbiased-strata front end on :8091, saved as JSON: run it against two engine builds and compare the files.
# After a numerics-touching change expect most answers identical and the rest to diverge only at a near-tie (fluent either way); see DEVLOG section 23.
import json,sys,time,urllib.request
PROMPTS=[
 "Explain in three sentences how a transistor works.",
 "Write a Python function that returns the n-th Fibonacci number, with a short docstring.",
 "What are the main causes of the French Revolution? Answer in a short paragraph.",
 "Translate to German: 'The weather is lovely today, so we are going for a walk by the river.'",
 "List five tips for learning a new language, one per line.",
 "Summarize the plot of Romeo and Juliet in four sentences.",
 "What is the derivative of x^3 * sin(x)? Show the steps briefly.",
 "Write a haiku about autumn rain, then explain its imagery in one sentence.",
 "Describe the difference between TCP and UDP in a few sentences.",
 "Give me a recipe for a simple tomato soup with quantities.",
]
out=[]
for p in PROMPTS:
    body={"model":"x","max_tokens":120,"temperature":0,"messages":[{"role":"user","content":p}],"reasoning_effort":"none"}
    req=urllib.request.Request('http://127.0.0.1:8091/v1/chat/completions',json.dumps(body).encode(),{'content-type':'application/json'})
    d=json.load(urllib.request.urlopen(req,timeout=900)); out.append(d['choices'][0]['message']['content'])
json.dump(out,open(sys.argv[1],'w'))
print("saved",len(out),"answers; avg chars",sum(map(len,out))//len(out))
