#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 pi05 -o 导出的两份动作 (fp16 为基准), 检查量化在板上的精度。
用法: python3 compare_actions.py <fp16的dump.bin> <量化模型的dump.bin> <对应eval目录>
示例: python3 compare_actions.py act_fp16_eval_s000.bin act_q80_eval_s000.bin ../q80_eval
兼容 Python 3.8, 仅依赖 numpy。
"""
import json
import sys

import numpy as np


def load(path):
    a = np.fromfile(path, dtype=np.float32)
    if a.size != 50 * 32:
        print('警告: %s 含 %d 个float, 期望1600 (=50步x32维, 含padding)' % (path, a.size))
    return a.reshape(-1, 32)[:, :16].astype(np.float64)  # 前16维为实际动作


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    ref_path, q_path, eval_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    ref, q = load(ref_path), load(q_path)

    for name, a in (('fp16 基准', ref), ('量化模型 ', q)):
        nan = int(np.isnan(a).sum())
        if nan:
            print('%s: 含 %d 个 NaN !!!' % (name, nan))
        else:
            print('%s: 无NaN, 范围 [%.3f, %.3f]' % (name, a.min(), a.max()))

    d = np.abs(q - ref)
    print('归一化空间: MAE=%.5f  max=%.4f  p99=%.4f' %
          (d.mean(), d.max(), np.quantile(d, 0.99)))

    with open(eval_dir.rstrip('/') + '/norm_stats.json') as f:
        ns = json.load(f)
    q01 = np.asarray(ns['action_q01'], dtype=np.float64)[:16]
    q99 = np.asarray(ns['action_q99'], dtype=np.float64)[:16]
    span = q99 - q01
    denorm = lambda x: ((x + 1.0) / 2.0) * span + q01
    dr = np.abs(denorm(q) - denorm(ref))
    print('物理单位:   R臂=%.4f rad  L臂=%.4f rad  R夹爪=%.2f  L夹爪=%.2f (0-100刻度)' %
          (dr[:, :7].mean(), dr[:, 8:15].mean(), dr[:, 7].mean(), dr[:, 15].mean()))
    print('            全维度 MAE=%.4f' % dr.mean())


if __name__ == '__main__':
    main()
