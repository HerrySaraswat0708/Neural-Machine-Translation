#!/bin/bash
# Round 2: capacity fix applied (configs/defaults.json now emb_dim=128,
# num_layers=6 -- proven-good scale from v1's word_sinusoidal run, replacing
# the ~100M-param config that collapsed to repeated-token degeneration for
# every tokenizer except char in round 1).
#
# Reduced matrix per user request: 3 tokenizers (bpe, word, char) x 2 PEs
# (sinusoidal, rope), attn=mha, 10 epochs. 6 runs, one per GPU, fully
# parallel -- encoded-tensor caches for all three tokenizers already exist
# from round 1 (encoding doesn't depend on model config), so no cache-race
# wave-staggering needed this time.
set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "=== Round 2: bpe/word/char x sinusoidal/rope, attn=mha, 10 epochs ==="
CUDA_VISIBLE_DEVICES=0 python -u train.py --tokenizer bpe  --pe sinusoidal --attn mha > logs/bpe_sinusoidal_mha.log  2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u train.py --tokenizer bpe  --pe rope       --attn mha > logs/bpe_rope_mha.log        2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u train.py --tokenizer word --pe sinusoidal --attn mha > logs/word_sinusoidal_mha.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u train.py --tokenizer word --pe rope       --attn mha > logs/word_rope_mha.log       2>&1 &
CUDA_VISIBLE_DEVICES=4 python -u train.py --tokenizer char --pe sinusoidal --attn mha > logs/char_sinusoidal_mha.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -u train.py --tokenizer char --pe rope       --attn mha > logs/char_rope_mha.log       2>&1 &
wait
echo "=== Round 2 training done: $(date) ==="

echo "=== Evaluating the 6 round-2 variants (greedy decode) ==="
for vid in bpe_sinusoidal_mha bpe_rope_mha word_sinusoidal_mha word_rope_mha char_sinusoidal_mha char_rope_mha; do
  python -u evaluate.py --variant "$vid" --decode greedy 2>&1 | tee -a logs/evaluate_round2.log
done

echo "=== DONE at $(date) ==="
