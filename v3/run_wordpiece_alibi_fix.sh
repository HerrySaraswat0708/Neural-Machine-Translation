#!/bin/bash
# Closes the last two "known gaps" in PROGRESS.md: WordPiece and ALiBi were
# only ever trained under the broken pre-fix oversized config (all
# collapsed to ~0 BLEU except char_alibi_mha, which survived by luck of
# char's tiny embedding table, not because ALiBi itself was competitive).
# Retraining these 6 cells under the correct capacity config completes the
# full 4-tokenizer x 3-PE breadth matrix (word/bpe/wordpiece/char x
# sinusoidal/rope/alibi) v3/README.md's Phase 4 actually promises -- the
# other 6 cells (word/bpe/char x sinusoidal/rope) are already done.
set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "=== WordPiece/ALiBi fix: 6 remaining breadth-matrix cells ==="
CUDA_VISIBLE_DEVICES=0 python -u train.py --tokenizer wordpiece --pe sinusoidal --attn mha > logs/wp_alibi_fix_wordpiece_sinusoidal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u train.py --tokenizer wordpiece --pe rope       --attn mha > logs/wp_alibi_fix_wordpiece_rope.log       2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u train.py --tokenizer wordpiece --pe alibi     --attn mha > logs/wp_alibi_fix_wordpiece_alibi.log      2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u train.py --tokenizer bpe       --pe alibi     --attn mha > logs/wp_alibi_fix_bpe_alibi.log            2>&1 &
CUDA_VISIBLE_DEVICES=4 python -u train.py --tokenizer word      --pe alibi     --attn mha > logs/wp_alibi_fix_word_alibi.log           2>&1 &
CUDA_VISIBLE_DEVICES=5 python -u train.py --tokenizer char      --pe alibi     --attn mha > logs/wp_alibi_fix_char_alibi.log           2>&1 &
wait
echo "=== WordPiece/ALiBi fix training done: $(date) ==="

echo "=== Evaluating the 6 variants (greedy decode) ==="
for vid in wordpiece_sinusoidal_mha wordpiece_rope_mha wordpiece_alibi_mha bpe_alibi_mha word_alibi_mha char_alibi_mha; do
  python -u evaluate.py --variant "$vid" --decode greedy 2>&1 | tee -a logs/evaluate_wp_alibi_fix.log
done

echo "=== WP ALIBI FIX DONE at $(date) ==="
