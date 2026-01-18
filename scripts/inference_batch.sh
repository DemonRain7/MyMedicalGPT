#!/bin/bash
# Batch Inference - 批量推理
# Process multiple queries from a file

python inference.py \
    --base_model merged-dpo \
    --template_name qwen \
    --data_file ./data/test_queries.jsonl \
    --output_file ./outputs/predictions.jsonl
