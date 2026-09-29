# Jetson Orin 部署测试清单

目标: 在 Jetson Orin (JetPack 5.1.3 / L4T R35.6.0 / CUDA 11.4) 上实测 fp16 / Q8_0 / Q4_K / IQ2_XS 四档的速度与精度。

## 1. 搬运内容 (共 ~18 GB)

### 模型 (大头, 每档 3 个文件)

| 服务器文件 | 板上目标路径 |
|---|---|
| `models/fp16_ft040000/pi05.gguf` (6.3G) | `<KIT>/fp16_eval/pi05.gguf` |
| `models/fp16_ft040000/pi05_q80_fisher.gguf` (4.6G) | `<KIT>/q80_eval/pi05.gguf` |
| `models/fp16_ft040000/pi05_q4k_fisher.gguf` (3.7G) | `<KIT>/q4k_eval/pi05.gguf` |
| `models/fp16_ft040000/pi05_iq2_xs_hsic_F_out_ft040000v2_smedian_lx1_0_ly1_0_bpw2_3471_Q4_K_fisher.gguf` (3.2G) | `<KIT>/iq2xs_eval/pi05.gguf` |

注: iq2xs 用 **ft040000v2** 那份 (摄像头掩码修复后的官方版)。每档目录另需:
`models/fp16_ft040000/norm_stats.json` + `models/fp16_ft040000/tokenizer.model` (共 4.1M, 各目录一份)。

### 测试素材 (小)
- `calib_g1/cmp_s{000,037,080,119,150,239}.png` → `<KIT>/test_imgs/`

### 脚本
- `jetson_kit/bench_jetson.sh` — 四档速度基准 (板上运行)
- `jetson_kit/compare_actions.py` — 动作 dump 精度对比 (板上运行, 需 numpy)
- `ActQuant/tools/pi0.5/main.cpp` — **唯一改动过的推理侧源码** (加了 `-o` 动作导出)

## 2. 搬运命令 (服务器上执行, J 改成板上地址)

```bash
J=<user@jetson-ip>
K=/home/hening/jetson_kit   # 板上的目录

ssh $J "mkdir -p $K/fp16_eval $K/q80_eval $K/q4k_eval $K/iq2xs_eval $K/test_imgs"
M=/home/hening/Quant_ws/pi05_libero_quant_eval
scp $M/models/fp16_ft040000/{norm_stats.json,tokenizer.model} $J:$K/fp16_eval/
scp $M/models/fp16_ft040000/{norm_stats.json,tokenizer.model} $J:$K/q80_eval/
scp $M/models/fp16_ft040000/{norm_stats.json,tokenizer.model} $J:$K/q4k_eval/
scp $M/models/fp16_ft040000/{norm_stats.json,tokenizer.model} $J:$K/iq2xs_eval/
scp $M/models/fp16_ft040000/pi05.gguf            $J:$K/fp16_eval/pi05.gguf
scp $M/models/fp16_ft040000/pi05_q80_fisher.gguf $J:$K/q80_eval/pi05.gguf
scp $M/models/fp16_ft040000/pi05_q4k_fisher.gguf $J:$K/q4k_eval/pi05.gguf
scp "$M/models/fp16_ft040000/pi05_iq2_xs_hsic_F_out_ft040000v2_smedian_lx1_0_ly1_0_bpw2_3471_Q4_K_fisher.gguf" $J:$K/iq2xs_eval/pi05.gguf
scp $M/calib_g1/cmp_s*.png $J:$K/test_imgs/
scp $M/jetson_kit/bench_jetson.sh $M/jetson_kit/compare_actions.py $J:$K/
scp /home/hening/Quant_ws/ActQuant/tools/pi0.5/main.cpp $J:$K/main.cpp
```

**不要搬**: amf_* imatrix (18G)、pali_llm_bf16.gguf (中间产物)、weights*/ 训练权重、量化用 Python 工具——量化只在服务器做, 板上只做推理。

## 3. 板上检查与运行

```bash
# 编译架构检查 (Orin 应为 87):
grep CMAKE_CUDA_ARCHITECTURES <build_dir>/CMakeCache.txt
# 确认二进制带 -o 选项 (服务器 9/28 加的补丁; 没有就用传过去的 main.cpp 重编):
./bin/pi05 --help 2>&1 | grep -- -o
# 缺的话: 用 main.cpp 覆盖 tools/pi0.5/main.cpp 后
#   cmake --build <build_dir> --target pi05 -j$(nproc)

# 锁频 (否则计时波动很大):
sudo nvpmodel -m 0 && sudo jetson_clocks

# 四档基准 (每样本 1 次预热 + 3 次计时, 自动汇总中位数):
bash ~/jetson_kit/bench_jetson.sh ~/jetson_kit 3

# 精度抽查 (每档跑一个样本, 与同板 fp16 比):
python3 ~/jetson_kit/compare_actions.py \
    ~/jetson_kit/bench_results/act_fp16_eval_s000.bin \
    ~/jetson_kit/bench_results/act_q80_eval_s000.bin ~/jetson_kit/q80_eval
```

## 4. 判读标准
- 速度: 板上 (带宽受限) 预期排序与服务器相反——位宽越低越快; Q8_0 应明显快于 fp16。
- 精度: 归一化 MAE 参考服务器实测 Q8_0≈0.002 / Q4_K≈0.011; 板上因内核不同数值略有差异, 同量级即可。
- 出现 NaN 或输出发散 → 立即停, 报告现象 (多半是 CUDA 版本/ggml 内核问题)。
