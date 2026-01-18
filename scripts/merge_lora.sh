#!/bin/bash
# Merge LoRA weights back to base model
# 合并LoRA权重到基座模型

# Merge PT model
echo "Merging PT model..."
python merge_peft_adapter.py \
    --base_model Qwen/Qwen2.5-0.5B \
    --lora_model outputs-pt \
    --output_dir merged-pt

# Merge SFT model
echo "Merging SFT model..."
python merge_peft_adapter.py \
    --base_model merged-pt \
    --lora_model outputs-sft \
    --output_dir merged-sft

# Merge DPO model
echo "Merging DPO model..."
python merge_peft_adapter.py \
    --base_model merged-sft \
    --lora_model outputs-dpo \
    --output_dir merged-dpo

echo "All models merged successfully!"
