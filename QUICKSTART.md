# 快速开始指南 - 5分钟上手

## 🎯 目标
这份指南帮助你在5分钟内完成第一次训练和推理。

## 步骤1: 环境准备 (1分钟)

```bash
# 克隆或复制MyMedicalGPT文件夹
cd MyMedicalGPT

# 安装依赖
pip install -r requirements.txt
```

## 步骤2: 准备数据 (可选)

项目已包含示例数据，位于 `data/` 目录：
- `data/pretrain/` - 预训练文本
- `data/finetune/` - SFT指令数据
- `data/reward/` - DPO偏好数据

如果要使用自己的数据，参考这些文件的格式。

## 步骤3: 只训练SFT (推荐新手)

如果你是第一次尝试，建议跳过PT阶段，直接做SFT：

```bash
# 使用Qwen2.5-0.5B基座模型直接做SFT
python supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --train_file_dir ./data/finetune \
    --validation_file_dir ./data/finetune \
    --per_device_train_batch_size 4 \
    --do_train \
    --use_peft True \
    --template_name qwen \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --output_dir outputs-sft \
    --bf16 \
    --target_modules all \
    --lora_rank 8 \
    --gradient_checkpointing True
```

这会在10-30分钟内完成（取决于你的GPU）。

## 步骤4: 合并模型

```bash
python merge_peft_adapter.py \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --lora_model outputs-sft \
    --output_dir my-first-model
```

## 步骤5: 测试推理

### 方式A: 命令行交互

```bash
python inference.py \
    --base_model my-first-model \
    --template_name qwen \
    --interactive
```

然后输入问题测试：
```
USER: 介绍一下人工智能
ASSISTANT: ...
```

### 方式B: Web界面 (更友好)

```bash
# 先安装gradio
pip install gradio

# 启动界面
python inference_gradio.py --model_path my-first-model
```

访问 http://localhost:7860

## 🎉 恭喜！

你已经完成了第一个模型的训练和推理！

## 下一步做什么？

### 1. 完整训练流程 (PT -> SFT -> DPO)

```bash
bash scripts/run_pipeline.sh
```

这会完整走一遍三个阶段。

### 2. 使用自己的数据

**SFT数据格式** (`data/finetune/my_data.jsonl`):
```json
{"conversations": [
  {"from": "human", "value": "问题1"},
  {"from": "gpt", "value": "回答1"}
]}
```

然后修改训练脚本的 `--train_file_dir`。

### 3. 调整训练参数

编辑 `scripts/train_sft.sh`，修改：
- `--num_train_epochs` - 训练轮数
- `--learning_rate` - 学习率
- `--lora_rank` - LoRA秩(越大效果越好，显存越大)

### 4. 部署到生产环境

参考 [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) 了解vLLM等高性能部署方案。

## 常见问题

### Q: 显存不够怎么办？
A: 减小batch size或使用量化：
```bash
python inference.py \
    --base_model my-first-model \
    --load_in_8bit  # 显存减半
```

### Q: 训练太慢？
A:
- 减小数据集: `--max_train_samples 1000`
- 减少epoch: `--num_train_epochs 1`
- 使用更小的模型

### Q: 模型效果不好？
A:
- 增加训练数据
- 提高训练轮数
- 检查数据质量
- 尝试更大的基座模型

### Q: 如何评估模型？
A:
```bash
# 在验证集上评估
python supervised_finetuning.py \
    --model_name_or_path my-first-model \
    --validation_file_dir ./data/finetune \
    --do_eval \
    --per_device_eval_batch_size 4
```

## 学习资源

- [README.md](README.md) - 完整项目文档
- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) - 推理技术详解

祝你训练愉快! 🚀
