# Jetson AGX Orin 复现指南（从零 clone 到基准）

在 Jetson AGX Orin (JetPack 5.1.3 / L4T R35.6.0 / CUDA 11.4) 上复现
fp16 / Q8_0 / Q4_K / IQ2_XS 四档的速度与精度基准。

已验证环境（2026-09-28 实测）：AGX Orin 64GB · JetPack 5.1.3 (L4T R35.6.0) ·
CUDA 11.4.315 · sm_87 · MAXN 锁频。

## 0. 预期结果（先看结论）

| 档位 | 内存 | 每进程基准* | 稳态/请求 | 归一化 MAE (vs fp16) |
|---|---|---|---|---|
| fp16 | 6.3 GB | 1259 ms | 365 ms | — |
| Q8_0 | 4.6 GB | 1253 ms | 372 ms | 0.002 |
| Q4_K | 3.7 GB | 1242 ms | 360 ms | 0.011 |
| IQ2_XS | 3.2 GB | 1340 ms | 382 ms | ≈0.047（劣化 4×，逼近发散） |

\* 每进程基准 = `bench_jetson.sh`，每轮新进程，**含 ~0.9 s 一次性初始化**。
稳态 = `steady_state_bench.py`，常驻进程多次推理取后 4 次中位 —— 部署形态
（serve_policy 常驻服务）的真实延迟。

**板上实测结论（修正了服务器端的预期）**：
- 量化在此 iGPU 上**速度中性**（LLM 侧是 kernel 效率瓶颈，不是带宽瓶颈），
  "位宽越低越快"在板上不成立；量化的收益是**内存**（6.3→3.7 GB）。
- 部署推荐：常驻服务 + **Q4_K**（360ms ≈ fp16，内存省 41%，MAE 0.011 可用）。
- fp16 冷启动首请求 ~1.3 s，常驻形态需预热一次。
- 全档输出无 NaN；所有量化档精度与服务器实测完全复现。

## 1. 编译（约 30 分钟量级）

上游 README 只在 x86 + CUDA 12.x 验证过；以下配置在 Jetson + CUDA 11.4 全绿。

```bash
# 1) Python 环境（py3.11；推理全在 C++ 侧，不需要 torch）
conda create -n actquant python=3.11 -y
conda activate actquant
pip install pybind11 numpy msgpack opencv-python-headless websockets huggingface_hub
# 版本硬约束：pybind11>=2.13.2（numpy2 C-API）、websockets>=13、huggingface_hub<1.0

# 2) 构建工具：系统 cmake 3.16 太老（ggml-cuda 需 >=3.18）
pip install cmake ninja    # 或用任一已有 conda env 里的 cmake>=3.18 + ninja

# 3) 解压 vendored 依赖（sentencepiece 源码在 zip 里，构建前必须解）
unzip vendor/tokenizers-cpp.zip -d vendor/

# 4) 配置 + 编译（在仓库根目录）
cmake -B build_openpi -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DGGML_CUDA_NO_VMM=ON \
    -DBUILD_PI05_PYTHON=ON \
    -DPython_EXECUTABLE=$(which python)
cmake --build build_openpi -j$(nproc)
```

**两个 Jetson 关键 flag，缺一不可：**
- `-DCMAKE_CUDA_ARCHITECTURES=87`：默认架构列表不含 87（Orin），不设会编出
  PTX-only 或直接失败。上游 README 写的 86 是给桌面 RTX 的。
- `-DGGML_CUDA_NO_VMM=ON`：Orin iGPU 驱动拒绝 VMM 显存池的 32GB VA 预留
  （`cuMemAddressReserve` 报 OOM，挂在动作专家图）。此 flag 回退 cudaMalloc 池。
  桌面独显不需要，但加上也无害。

产物在 `build_openpi/bin/`：`pi05`（CLI）、`pi05.so`（Python 绑定）、
`llama-quantize` 等。**build 目录不能挪**——二进制 RUNPATH 是绝对路径。

其它 GPU 型号：Orin 家族以外的设备改 arch（如桌面 RTX 用 86/89）并去掉
`GGML_CUDA_NO_VMM`；速度数字只在同级 AGX Orin 上可复现。

## 2. 权重与素材（~18 GB，不入 git）

仓库不含 GGUF 权重（`.gitignore` 排除，且是内部微调权重）。从服务器搬运：

| 服务器文件 | 板上目标路径 |
|---|---|
| `models/fp16_ft040000/pi05.gguf` (6.3G) | `<KIT>/fp16_eval/pi05.gguf` |
| `models/fp16_ft040000/pi05_q80_fisher.gguf` (4.6G) | `<KIT>/q80_eval/pi05.gguf` |
| `models/fp16_ft040000/pi05_q4k_fisher.gguf` (3.7G) | `<KIT>/q4k_eval/pi05.gguf` |
| `models/fp16_ft040000/pi05_iq2_xs_hsic_F_out_ft040000v2_smedian_lx1_0_ly1_0_bpw2_3471_Q4_K_fisher.gguf` (3.2G) | `<KIT>/iq2xs_eval/pi05.gguf` |

注: iq2xs 用 **ft040000v2** 那份（摄像头掩码修复后的官方版）。每档目录另需
`norm_stats.json` + `tokenizer.model`（共 4.1M，各目录一份）；测试图
`calib_g1/cmp_s{000,037,080,119,150,239}.png` → `<KIT>/test_imgs/`。
目录布局即本目录（`acquant_gguf/`）的布局。

**不要搬**: amf_* imatrix (18G)、pali_llm_bf16.gguf（中间产物）、训练权重、
量化用 Python 工具——量化只在服务器做，板上只做推理。

## 3. 速度基准

```bash
# 锁频（否则计时波动很大，EMC 不锁会吃掉低比特带宽红利）：
sudo nvpmodel -m 0 && sudo jetson_clocks

# 口径 A：每进程基准（含一次性初始化，衡量冷启动）
bash acquant_gguf/bench_jetson.sh acquant_gguf 3
#   每样本 1 次预热 + 3 次计时，自动汇总中位数；二进制路径自动按仓库推导，
#   可用 BIN=/path/to/pi05 覆盖。预期 ~1240-1340 ms（大头是 ~0.9s 进程初始化）。

# 口径 B：稳态延迟（常驻进程，衡量部署）
python acquant_gguf/steady_state_bench.py   # [kit_dir] [bin_dir] [tiers...] 可选
#   预期 360-382 ms（见第 0 节表）。
```

**CUDA 图开关（推荐打开）**：

```bash
export GGML_CUDA_GRAPH_ALLOW_BATCH=1   # 放行 pi0.5 的 batch>1 ADD 节点
export GGML_CUDA_GRAPH_PROBE=1         # 可选：打印每个图能否 capture 及 bail 原因
```

pi0.5 的图形状完全静态，ggml 的 re-capture 机制保证了正确性；开关只影响
上游为动态 batch 写的保守 bail 规则。已验证 4 图全 capture 且输出与 eager
**逐字节一致**。不开也能跑（功能正常），稳态慢 ~10 ms、kernel 启动数
12k→2.9k 的差别。默认不开启（保持上游行为）。

## 4. 精度抽查

```bash
# bench_jetson.sh 已导出各档动作 dump（bench_results/act_*.bin，git 已忽略）：
python acquant_gguf/compare_actions.py \
    acquant_gguf/bench_results/act_fp16_eval_s000.bin \
    acquant_gguf/bench_results/act_q80_eval_s000.bin acquant_gguf/q80_eval
```

判读标准（归一化 MAE，6 样本均值）：Q8_0≈0.002 / Q4_K≈0.011 / IQ2_XS≈0.047。
板上与服务器数值应同量级（内核不同略有差异）。出现 NaN 或输出发散 →
立即停，报告现象（多半是 CUDA 版本 / ggml 内核问题）。
