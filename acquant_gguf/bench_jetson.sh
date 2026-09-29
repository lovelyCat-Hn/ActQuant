#!/bin/bash
# Jetson Orin: fp16 / Q8_0 / Q4_K / IQ2_XS 四档速度基准
# 用法: bash bench_jetson.sh <模型根目录> [每组计时轮数,默认3]
#   模型布局: <根目录>/{fp16_eval,q80_eval,q4k_eval,iq2xs_eval}/
#             各含 pi05.gguf + norm_stats.json + tokenizer.model
#             以及 <根目录>/test_imgs/cmp_s*.png
#   二进制路径默认 $HOME/Quant_ws/ActQuant/build_openpi/bin/pi05
#   可用环境变量覆盖: BIN=/path/to/pi05 bash bench_jetson.sh ...
# 计时建议(否则波动大):
#   sudo nvpmodel -m 0 && sudo jetson_clocks
set -e

MODELS=${1:?用法: bash bench_jetson.sh <模型根目录> [轮数]}
N=${2:-3}
BIN=${BIN:-$HOME/Quant_ws/ActQuant/build_openpi/bin/pi05}
IMGS=$MODELS/test_imgs
OUT=$MODELS/bench_results
mkdir -p "$OUT"

PROMPT_PICK="left arm pick up a. right arm pick up a."
PROMPT_PLACE="left arm places a in the top-left corner. right arm places a in the top-left corner."
SAMPLES="000 037 080 119 150 239"

RESULT=$OUT/times.txt
: > "$RESULT"

for V in fp16_eval q80_eval q4k_eval iq2xs_eval; do
  DIR=$MODELS/$V
  [ -d "$DIR" ] || { echo "跳过 $V (目录不存在)"; continue; }
  echo "=== $V ==="
  for s in $SAMPLES; do
    IMG=$IMGS/cmp_s${s}.png
    [ -f "$IMG" ] || { echo "缺 $IMG, 跳过"; continue; }
    if [ "$s" = "150" ] || [ "$s" = "239" ]; then P="$PROMPT_PLACE"; else P="$PROMPT_PICK"; fi
    # 1 次预热 (不计入)
    CUDA_VISIBLE_DEVICES=0 "$BIN" -m "$DIR" -i "$IMG" -p "$P" -d cuda \
        > /dev/null 2>&1 || { echo "sample $s 预热失败!"; continue; }
    # N 次计时, 最后一次导出动作
    for i in $(seq 1 "$N"); do
      LOG=$OUT/run_${V}_s${s}_${i}.txt
      CUDA_VISIBLE_DEVICES=0 "$BIN" -m "$DIR" -i "$IMG" -p "$P" -d cuda \
          -o "$OUT/act_${V}_s${s}.bin" > "$LOG" 2>&1
      T=$(grep -oE 'Total inference time: [0-9.]+ ms' "$LOG" | grep -oE '[0-9.]+')
      echo "$V $s $T" >> "$RESULT"
      echo "  sample $s run $i: ${T} ms"
    done
  done
done

echo ""
echo "==== 汇总 (中位数 / 最小值, ms) ===="
awk '{a[$1]=a[$1]" "$3} END {
  for (v in a) {
    n=split(a[v], t, " ")
    for(i=1;i<=n;i++) vals[i]=t[i]+0
    for(i=1;i<=n;i++) for(j=i+1;j<=n;j++) if(vals[j]<vals[i]) {tmp=vals[i];vals[i]=vals[j];vals[j]=tmp}
    med = (n%2) ? vals[(n+1)/2] : (vals[n/2]+vals[n/2+1])/2
    printf "%-10s 中位=%.1f  最小=%.1f  (n=%d)\n", v, med, vals[1], n
  }
}' "$RESULT" | sort
echo "明细: $RESULT"
