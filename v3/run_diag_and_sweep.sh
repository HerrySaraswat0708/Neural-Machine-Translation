#!/bin/bash
# Step 1: diagnose why char_sinusoidal_mha collapses at greedy-decode time
# despite training/eval-loss being healthy (69% eval token-acc).
# Step 2: decoding-strategy sweep (beam / top_k / top_p) for the winning
# config, bpe_rope_mha, to pick the best decoding config on top of it.
# Runs fully detached in tmux so it survives the client terminal closing.
set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "=== [1/2] char_sinusoidal_mha generation diagnostic ==="
CUDA_VISIBLE_DEVICES=0 python3 -u probe_char_sinusoidal.py > logs/char_sinusoidal_probe.log 2>&1
echo "--- probe output ---"
cat logs/char_sinusoidal_probe.log
echo "=== probe done: $(date) ==="

echo "=== [2/2] decoding-strategy sweep for bpe_rope_mha (beam/top_k/top_p) ==="
CUDA_VISIBLE_DEVICES=0 python -u evaluate.py --variant bpe_rope_mha --decode beam  > logs/bpe_rope_mha_beam.log  2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u evaluate.py --variant bpe_rope_mha --decode top_k > logs/bpe_rope_mha_topk.log  2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u evaluate.py --variant bpe_rope_mha --decode top_p > logs/bpe_rope_mha_topp.log  2>&1 &
wait
echo "=== decoding sweep done: $(date) ==="

echo "=== ALL DONE at $(date) ==="
