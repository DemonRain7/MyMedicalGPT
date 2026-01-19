# MyMedicalGPT - 精简版LLM训练框架

一个精简的大语言模型训练框架，包含完整的训练和推理流程。

## 📁 项目结构

```
MyMedicalGPT/
├── README.md                    # 项目文档
├── requirements.txt             # 依赖包
├── pretraining.py              # 预训练脚本
├── supervised_finetuning.py    # SFT微调脚本
├── dpo_training.py             # DPO训练脚本
├── merge_peft_adapter.py       # LoRA合并脚本
├── template.py                 # 对话模板
├── inference.py                # 基础推理
├── inference_api.py            # API推理服务
├── inference_gradio.py         # Gradio Web界面
├── inference_vllm.py           # vLLM高性能推理
├── scripts/                    # 训练脚本
│   ├── train_pt.sh            # 预训练
│   ├── train_sft.sh           # SFT微调
│   ├── train_dpo.sh           # DPO训练
│   ├── merge_lora.sh          # 合并LoRA
│   ├── run_pipeline.sh        # 完整流程
│   ├── inference_basic.sh     # 基础推理
│   ├── inference_batch.sh     # 批量推理
│   └── serve_vllm.sh          # vLLM服务
├── data/                       # 数据目录
│   ├── pretrain/              # 预训练数据(.txt)
│   ├── finetune/              # SFT数据(.jsonl)
│   └── reward/                # DPO数据(.jsonl)
├── configs/                    # 配置文件
└── notebooks/                  # Jupyter notebooks
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行完整训练流程

```bash
bash scripts/run_pipeline.sh
```

这会依次执行：
1. **Stage 1: 增量预训练 (PT)** - 在领域文本上继续训练
2. **Stage 2: 有监督微调 (SFT)** - 使用指令数据对齐
3. **Stage 3: 直接偏好优化 (DPO)** - 从人类偏好学习

### 3. 单独运行某个阶段

```bash
# 只运行预训练
bash scripts/train_pt.sh

# 只运行SFT
bash scripts/train_sft.sh

# 只运行DPO
bash scripts/train_dpo.sh
```

## 💾 数据格式

### 预训练数据 (.txt)
纯文本文件，一行一段或一篇文档：
```
这是第一段文本内容
这是第二段文本内容
```

### SFT数据 (.jsonl)
ShareGPT格式，每行一个JSON对象：
```json
{
  "conversations": [
    {"from": "human", "value": "你好"},
    {"from": "gpt", "value": "你好！有什么我可以帮助你的吗？"}
  ]
}
```

### DPO数据 (.jsonl)
偏好对比格式：
```json
{
  "system": "",
  "question": "什么是人工智能？",
  "response_chosen": "人工智能(AI)是计算机科学的一个分支...",
  "response_rejected": "AI就是机器人。"
}
```

## 🎯 推理使用

### 方式1: 命令行交互

```bash
python inference.py \
    --base_model merged-dpo \
    --template_name qwen \
    --interactive
```

### 方式2: FastAPI服务

```bash
# 启动API服务
python inference_api.py --model_path merged-dpo --port 8000

# 使用curl测试
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "介绍一下人工智能",
    "history": []
  }'
```

### 方式3: Gradio Web界面

```bash
python inference_gradio.py --model_path merged-dpo --port 7860
```

然后访问 http://localhost:7860

### 方式4: 批量推理

```bash
# 准备输入文件 queries.jsonl
echo '{"query": "什么是机器学习?"}' > queries.jsonl
echo '{"query": "深度学习有哪些应用?"}' >> queries.jsonl

# 批量推理
python inference.py \
    --base_model merged-dpo \
    --data_file queries.jsonl \
    --output_file predictions.jsonl
```

### 方式5: vLLM 高性能推理 (推荐生产环境)

vLLM 提供 10-20x 的吞吐量提升，适合生产部署。

```bash
# 安装 vLLM (需要 Linux + NVIDIA GPU)
pip install vllm

# 方式A: 交互式对话
python inference_vllm.py --model_path merged-dpo --interactive

# 方式B: 启动 OpenAI 兼容 API 服务
python inference_vllm.py --model_path merged-dpo --serve --port 8000

# 方式C: 批量推理
python inference_vllm.py \
    --model_path merged-dpo \
    --data_file queries.jsonl \
    --output_file vllm_output.jsonl
```

**调用 vLLM API 服务:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="merged-dpo",
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)
```

**vLLM 高级配置:**

```bash
# 多卡并行
python inference_vllm.py --model_path merged-dpo --serve \
    --tensor_parallel_size 2

# AWQ 量化
python inference_vllm.py --model_path merged-dpo --serve \
    --quantization awq

# 调整显存使用
python inference_vllm.py --model_path merged-dpo --serve \
    --gpu_memory_utilization 0.8
```

## ⚙️ 训练参数说明

### 核心参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--model_name_or_path` | 基座模型路径 | `Qwen/Qwen2.5-0.5B` |
| `--use_peft` | 是否使用LoRA | `True` |
| `--lora_rank` | LoRA秩 | 8-64 |
| `--learning_rate` | 学习率 | PT: 2e-4, SFT: 2e-5 |
| `--num_train_epochs` | 训练轮数 | 1-3 |
| `--per_device_train_batch_size` | 批次大小 | 2-8 |
| `--gradient_accumulation_steps` | 梯度累积 | 4-8 |

### 显存优化

- **使用LoRA**: `--use_peft True` (显存降低80%)
- **梯度检查点**: `--gradient_checkpointing True` (显存降低30%)
- **减小batch size**: `--per_device_train_batch_size 2`
- **增加梯度累积**: `--gradient_accumulation_steps 8`
- **使用bf16**: `--bf16 True`

## 📊 模型评估

训练日志保存在 `outputs-*/runs/`，使用TensorBoard查看：

```bash
tensorboard --logdir outputs-sft/runs --port 6006
```

## 🔧 常见问题

### 1. 显存不足
- 减小 `batch_size`
- 增加 `gradient_accumulation_steps`
- 使用 `--gradient_checkpointing`
- 减小 `block_size` 或 `max_length`

### 2. 训练速度慢
- 增大 `batch_size`
- 减少 `gradient_accumulation_steps`
- 关闭 `--gradient_checkpointing`
- 使用多卡训练

### 3. 模型效果差
- 增加训练数据量
- 调整学习率
- 增加训练轮数
- 检查数据质量

## 📝 自定义修改指南

### 添加自己的数据
1. 将数据转换为对应格式
2. 放入 `data/` 对应目录
3. 修改训练脚本中的 `--train_file_dir`

### 更换基座模型
1. 修改 `--model_name_or_path`
2. 调整 `--template_name` (vicuna/alpaca/qwen等)
3. 根据模型调整 `--target_modules`

### 调整训练策略
编辑 `scripts/train_*.sh`，修改参数：
- 学习率
- batch size
- 训练轮数
- LoRA配置

## 🎓 学习资源

- [HuggingFace Transformers文档](https://huggingface.co/docs/transformers)
- [PEFT (LoRA) 文档](https://huggingface.co/docs/peft)
- [TRL (RLHF) 文档](https://huggingface.co/docs/trl)

## 📄 许可证

本项目遵循 Apache 2.0 许可证。
