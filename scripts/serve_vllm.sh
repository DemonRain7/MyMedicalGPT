#!/bin/bash
# vLLM 高性能推理服务启动脚本
# 需要先安装 vLLM: pip install vllm

# 模型路径（修改为你的模型路径）
MODEL_PATH="merged-sft"

# 服务配置
HOST="0.0.0.0"
PORT=8000

# GPU 配置
TENSOR_PARALLEL_SIZE=1  # 多卡时增加
GPU_MEMORY_UTILIZATION=0.9  # 显存使用率

# 模型配置
MAX_MODEL_LEN=4096
DTYPE="auto"  # auto, float16, bfloat16

# 量化配置（可选）
# QUANTIZATION="awq"  # awq, gptq, squeezellm

echo "=============================================="
echo "Starting vLLM OpenAI API Server"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Server: http://$HOST:$PORT"
echo "=============================================="

# 方式1: 使用 inference_vllm.py 脚本
python inference_vllm.py \
    --model_path $MODEL_PATH \
    --serve \
    --host $HOST \
    --port $PORT \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
    --max_model_len $MAX_MODEL_LEN \
    --dtype $DTYPE

# 方式2: 直接使用 vLLM 命令（如果上面的方式不工作）
# python -m vllm.entrypoints.openai.api_server \
#     --model $MODEL_PATH \
#     --host $HOST \
#     --port $PORT \
#     --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
#     --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
#     --max-model-len $MAX_MODEL_LEN \
#     --dtype $DTYPE \
#     --trust-remote-code

# 服务启动后，可以使用以下方式调用:
#
# 1. curl 命令:
# curl http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "merged-sft",
#     "messages": [{"role": "user", "content": "你好"}]
#   }'
#
# 2. Python OpenAI SDK:
# from openai import OpenAI
# client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
# response = client.chat.completions.create(
#     model="merged-sft",
#     messages=[{"role": "user", "content": "你好"}]
# )
# print(response.choices[0].message.content)
