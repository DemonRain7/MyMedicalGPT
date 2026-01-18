#!/bin/bash
# Basic Inference - 基础推理
# Interactive mode for chatting with the model

python inference.py \
    --base_model merged-dpo \
    --template_name qwen \
    --interactive
