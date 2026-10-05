#!/bin/bash
# Architecture/hyperparameter tuning round 1 on the winning bpe+RoPE+MHA
# config (39.21 BLEU baseline, no rerun needed -- already in manifest as
# bpe_rope_mha). 7 targeted single/paired-axis changes, one per GPU, kept
# modest per the round-1 lesson (don't outsize the 106,531-pair dataset
# again). Requires train.py's new --emb_dim/--num_layers/--ff_dim/--dropout/
# --tag CLI overrides (added this session) so parallel architecture variants
# at the same tokenizer/pe/attn don't collide on checkpoint/manifest keys.
set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "=== Tuning round 1: 7 architecture/hparam variants on bpe+rope+mha ==="
CUDA_VISIBLE_DEVICES=7 python -u train.py --tokenizer bpe --pe rope --attn mha --num_layers 8                        --tag deeper     > logs/tune_deeper.log     2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u train.py --tokenizer bpe --pe rope --attn mha --emb_dim 192                         --tag wider      > logs/tune_wider.log      2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u train.py --tokenizer bpe --pe rope --attn mha --emb_dim 192 --num_layers 8          --tag widedeep   > logs/tune_widedeep.log   2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u train.py --tokenizer bpe --pe rope --attn mha --ff_dim 512                          --tag narrowffn  > logs/tune_narrowffn.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python -u train.py --tokenizer bpe --pe rope --attn mha --epochs 20                           --tag longtrain  > logs/tune_longtrain.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -u train.py --tokenizer bpe --pe rope --attn mha --dropout 0.05                        --tag lowdrop    > logs/tune_lowdrop.log   2>&1 &
CUDA_VISIBLE_DEVICES=6 python -u train.py --tokenizer bpe --pe rope --attn gqa                                                        > logs/tune_gqa.log       2>&1 &
wait
echo "=== Tuning round 1 training done: $(date) ==="

echo "=== Evaluating the 7 tuning variants (greedy decode) ==="
for vid in bpe_rope_mha_deeper bpe_rope_mha_wider bpe_rope_mha_widedeep bpe_rope_mha_narrowffn bpe_rope_mha_longtrain bpe_rope_mha_lowdrop bpe_rope_gqa; do
  python -u evaluate.py --variant "$vid" --decode greedy 2>&1 | tee -a logs/evaluate_tune1.log
done

echo "=== TUNE1 DONE at $(date) ==="
