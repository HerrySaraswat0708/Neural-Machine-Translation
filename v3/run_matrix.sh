#!/bin/bash
# Runs the breadth matrix (12: tokenizer x PE, attn=mha) + attention matrix
# (3: mha/mqa/gqa at bpe+rope -- the mha row is reused from the breadth
# matrix's bpe_rope_mha run rather than retrained) across all visible GPUs.
#
# Waved instead of one flat parallel launch because vocab + encoded-tensor
# caching (train.py) is keyed by tokenizer only and built lazily on first
# use -- two jobs sharing a tokenizer starting at the same instant would
# race on writing that cache. Wave 1 trains exactly one job per tokenizer
# first (also 4 of the 12 breadth cells), so every tokenizer's cache is
# warm and read-only before any parallel wave that reuses it.
set -e
cd "$(dirname "$0")"
mkdir -p logs

GPUS=(0 1 2 3 4 5 6 7)
NGPU=${#GPUS[@]}

run_variant() {
  local tok=$1 pe=$2 attn=$3 gpu=$4
  local vid="${tok}_${pe}_${attn}"
  local ckpt="checkpoints/${vid}/model.pt"
  if [ -f "$ckpt" ]; then
    echo ">>> Skipping ${vid}: checkpoint already exists"
    return
  fi
  echo ">>> Launching ${vid} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES=$gpu python -u train.py --tokenizer "$tok" --pe "$pe" --attn "$attn" \
    > "logs/${vid}.log" 2>&1 &
}

echo "=== Wave 1: one job per tokenizer (word, bpe, wordpiece, char), pe=sinusoidal, attn=mha ==="
echo "    -- builds vocab + encoded-tensor cache for each tokenizer as a side effect"
wave1=(word bpe wordpiece char)
for i in "${!wave1[@]}"; do
  run_variant "${wave1[$i]}" sinusoidal mha "${GPUS[$((i % NGPU))]}"
done
wait
echo "=== Wave 1 done: $(date) ==="

echo "=== Wave 2: remaining 8 breadth cells (rope/alibi x word/bpe/wordpiece/char), attn=mha ==="
wave2=(
  "word rope" "word alibi"
  "bpe rope" "bpe alibi"
  "wordpiece rope" "wordpiece alibi"
  "char rope" "char alibi"
)
for i in "${!wave2[@]}"; do
  read -r tok pe <<< "${wave2[$i]}"
  run_variant "$tok" "$pe" mha "${GPUS[$((i % NGPU))]}"
done
wait
echo "=== Wave 2 done: $(date) ==="

echo "=== Wave 3: attention matrix (bpe+rope, attn=mqa/gqa -- mha row reused from bpe_rope_mha above) ==="
run_variant bpe rope mqa "${GPUS[0]}"
run_variant bpe rope gqa "${GPUS[1]}"
wait
echo "=== Wave 3 done: $(date) ==="

echo "=== All 14 unique training runs complete (12 breadth cells + 2 new attention cells; ==="
echo "=== bpe_rope_mha serves as the attention matrix's third, shared row). Evaluating... ==="
python -u evaluate.py --all --decode greedy 2>&1 | tee -a logs/evaluate_all.log

echo "=== DONE at $(date) ==="
