#!/bin/bash
# Complete Training Pipeline: PT -> SFT -> DPO
# 完整训练流程

set -e  # Exit on error

echo "========================================="
echo "Starting Complete Training Pipeline"
echo "========================================="

# Stage 1: Pretraining
echo ""
echo "Stage 1/3: Continue Pretraining..."
bash scripts/train_pt.sh

# Merge PT model
echo ""
echo "Merging PT model..."
python merge_peft_adapter.py \
    --base_model Qwen/Qwen2.5-0.5B \
    --lora_model outputs-pt \
    --output_dir merged-pt

# Stage 2: Supervised Fine-Tuning
echo ""
echo "Stage 2/3: Supervised Fine-Tuning..."
bash scripts/train_sft.sh

# Merge SFT model
echo ""
echo "Merging SFT model..."
python merge_peft_adapter.py \
    --base_model merged-pt \
    --lora_model outputs-sft \
    --output_dir merged-sft

# Stage 3: DPO
echo ""
echo "Stage 3/3: Direct Preference Optimization..."
bash scripts/train_dpo.sh

# Merge DPO model
echo ""
echo "Merging DPO model..."
python merge_peft_adapter.py \
    --base_model merged-sft \
    --lora_model outputs-dpo \
    --output_dir merged-dpo

echo ""
echo "========================================="
echo "Training Pipeline Completed!"
echo "Final model: ./merged-dpo"
echo "========================================="
