# SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
# SPDX-License-Identifier: MIT
import sys,random
n=int(sys.argv[1]); random.seed(7)
towns="Aldoran Brevik Cantor Dunmere Eskel Falkor Garrow Hollin Ivar Jessup Korvath Lundy Maren Nesso Orrin Pellam".split()
lines=[f"Fact {i}: the town of {random.choice(towns)}{i} has {random.randint(2,97)} lighthouses and {random.randint(11,950)} boats." for i in range(1,n+1)]
k=max(1,n//2)
print("<|im_start|>user\n"+"\n".join(lines)+f"\n\nWhat is Fact {k}? Repeat it exactly.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",end="")
print(lines[k-1],file=sys.stderr)
