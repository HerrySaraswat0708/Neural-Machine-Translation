#!/bin/bash
# Trains all 6 NMT variants sequentially, then evaluates and saves results.
set -e
cd /home/herrys/projects/Neural-Machine-Translation

LOG=run_all.log
echo "=== NMT full training run started at $(date) ===" | tee $LOG

variants=(
  "word sinusoidal"
  "word rope"
  "bpe sinusoidal"
  "bpe rope"
  "char sinusoidal"
  "char rope"
)

for variant in "${variants[@]}"; do
  tok=$(echo $variant | awk '{print $1}')
  pe=$(echo $variant | awk '{print $2}')
  ckpt="checkpoints/${tok}_${pe}/model.pt"
  in_manifest=$(python3 -c "import manifest; print('yes' if '${tok}_${pe}' in manifest.load() else 'no')" 2>/dev/null)
  if [ -f "$ckpt" ] && [ "$in_manifest" = "yes" ]; then
    echo "" | tee -a $LOG
    echo ">>> Skipping ${tok}_${pe}: checkpoint + manifest entry already exist" | tee -a $LOG
    continue
  fi
  echo "" | tee -a $LOG
  echo ">>> Training: tokenizer=$tok  pe=$pe  started at $(date)" | tee -a $LOG
  python -u train.py --tokenizer $tok --pe $pe 2>&1 | tee -a $LOG
  echo ">>> Done: tokenizer=$tok  pe=$pe  finished at $(date)" | tee -a $LOG
done

echo "" | tee -a $LOG
echo "=== All variants trained. Running evaluation ===" | tee -a $LOG
python -u evaluate.py --all --all-decodes 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== DONE at $(date) ===" | tee -a $LOG
echo "Launch the app with: streamlit run app.py"
