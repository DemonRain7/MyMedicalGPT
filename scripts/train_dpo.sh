#!/bin/bash
# Stage 3: Direct Preference Optimization
# 直接偏好优化 - 从人类偏好数据中学习

python dpo_training.py \
    --model_name_or_path ./merged-sft \
    --template_name qwen \
    --train_file_dir ./data/reward \
    --validation_file_dir ./data/reward \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples 1000 \
    --max_eval_samples 100 \
    --max_steps 200 \
    --eval_steps 20 \
    --save_steps 100 \
    --max_source_length 512 \
    --max_target_length 512 \
    --output_dir outputs-dpo \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 True \
    --fp16 False \
    --device_map auto \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --cache_dir ./cache
