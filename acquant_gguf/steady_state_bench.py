#!/usr/bin/env python3
"""Steady-state (persistent-process) latency benchmark for the pi0.5 quant tiers.

bench_jetson.sh spawns a fresh process per run, so every measurement includes
the ~0.9 s one-time per-process init (cudaMalloc / cuBLAS workspace storm).
This script loads each tier ONCE and times repeated inferences — the number a
persistent server (serve_policy.py) actually delivers. The two methodologies
answer different questions; report both.

Usage:
    python3 steady_state_bench.py [kit_dir] [build_bin_dir] [tiers...]
    kit_dir      dir containing fp16_eval/ q80_eval/ ... and test_imgs/
                 (default: the directory containing this script)
    build_bin_dir dir with pi05.so + libggml*.so
                 (default: <repo>/build_openpi/bin)

Methodology note: the FIRST inference of the whole process pays init (~+0.9 s);
within each tier the first two calls still settle lazily-allocated work, so the
steady-state figure is the median of the last 4 of 6 runs. This reproduces the
2026-09-28 measurements (AGX Orin, MAXN locked): fp16 365 / Q8_0 372 / Q4_K 360
/ IQ2_XS 382 ms.
"""
import os
import sys
import time

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
kit = sys.argv[1] if len(sys.argv) > 1 else here
bin_dir = (sys.argv[2] if len(sys.argv) > 2
           else os.path.join(os.path.dirname(here), 'build_openpi', 'bin'))
tiers = sys.argv[3:] or ['fp16_eval', 'q80_eval', 'q4k_eval', 'iq2xs_eval']

sys.path.insert(0, bin_dir)
import pi05  # noqa: E402

PROMPT = "left arm pick up a. right arm pick up a."
IMGS = ['cmp_s000.png', 'cmp_s037.png', 'cmp_s080.png',
        'cmp_s119.png', 'cmp_s150.png', 'cmp_s239.png']

for tier in tiers:
    d = os.path.join(kit, tier)
    p = pi05.Pi05Pipeline(os.path.join(d, 'pi05.gguf'),
                          os.path.join(d, 'tokenizer.model'), 'cuda', 4, 10)
    times = []
    for img in IMGS:
        t = time.perf_counter()
        np.asarray(p.run(os.path.join(kit, 'test_imgs', img), PROMPT))
        times.append((time.perf_counter() - t) * 1000)
    steady = sorted(times[2:])  # first 2 settle lazily-allocated work
    med = steady[len(steady) // 2]
    print(f"{tier:12s} first={times[0]:7.1f} ms  "
          f"steady(last4)={[f'{x:.0f}' for x in times[2:]]}  median={med:6.1f} ms",
          flush=True)
    del p
