#!/bin/bash
#SBATCH --partition=gpu-l40s-low
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=train_swinunet_%j.log

set -euo pipefail

source /mnt/fastscratch/users/sghjia14/init_env.sh
cd /mnt/fastscratch/users/sghjia14/unet

echo "========================================"
echo "Job started : $(date)"
echo "Node        : ${SLURM_NODELIST:-N/A}"
echo "Job ID      : ${SLURM_JOB_ID:-N/A}"
echo "Config      : Case1 + Pretrained (α=0.5, β=0.5, w=[0.05,1.5,4.0])"
python - <<'PY'
import torch, timm
print('torch:', torch.__version__)
print('timm:', timm.__version__)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
echo "========================================"

mkdir -p checkpoints_swinunet


python train_swinunet.py \
  --pretrained \
  --epochs 50 \
  --batch-size 8 \
  --lr 5e-5 \
  --model-name swinv2_small_window16_256.ms_in1k \
  --decoder-channels 256 128 64 32 \
  --target-size 512 \
  --class-filter allow-missing \
  --loss focal_tversky \
  --tversky-alpha 0.5 \
  --tversky-beta 0.5 \
  --focal-tversky-gamma 1.33 \
  --class-weights 0.05 1.5 4.0 \
  2>&1 | tee "train_swinunet_case1_pretrained.log"

echo "========================================"
echo "Job finished: $(date)"
echo "========================================"

LOG_FILE="train_swinunet_${SLURM_JOB_ID}.log"
DATE_STR=$(date +%Y%m%d)
CKPT_DIR=$(ls -dt checkpoints_swinunet/${DATE_STR}* 2>/dev/null | head -1 || true)
[ -n "${CKPT_DIR}" ] && mv "${LOG_FILE}" "${CKPT_DIR}/" 2>/dev/null || true