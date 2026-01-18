# 推理技术详解 - Inference Techniques Guide

这份文档详细介绍了各种LLM推理优化技术，以及如何在MyMedicalGPT中应用它们。

## 📚 目录

1. [基础推理方法](#1-基础推理方法)
2. [高级推理优化](#2-高级推理优化)
3. [部署方案](#3-部署方案)
4. [性能对比](#4-性能对比)

---

## 1. 基础推理方法

### 1.1 直接推理 (Naive Inference)

**原理**: 最简单的方式，加载完整模型到GPU/CPU直接生成。

**使用场景**:
- 开发测试
- 小模型(<7B)
- 单用户使用

**代码示例**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("merged-dpo")
tokenizer = AutoTokenizer.from_pretrained("merged-dpo")

inputs = tokenizer("你好", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

**优缺点**:
- ✅ 实现简单
- ❌ 显存占用大
- ❌ 吞吐量低
- ❌ 不支持并发

---

### 1.2 批量推理 (Batch Inference)

**原理**: 将多个请求打包成batch一起处理，提高GPU利用率。

**使用场景**:
- 离线评估
- 数据标注
- 批量翻译

**代码示例**:
```python
# 见 scripts/inference_batch.sh
python inference.py \
    --base_model merged-dpo \
    --data_file queries.jsonl \
    --output_file predictions.jsonl
```

**优化技巧**:
- 使用 `DataLoader` 自动padding
- 设置合适的 `batch_size`
- 启用 `torch.compile()` (PyTorch 2.0+)

**优缺点**:
- ✅ 吞吐量高
- ✅ 成本低
- ❌ 延迟高
- ❌ 不适合实时场景

---

## 2. 高级推理优化

### 2.1 量化 (Quantization)

**原理**: 将模型权重从FP16/BF16降低到INT8/INT4，减少显存和计算量。

#### 2.1.1 动态量化 (Post-Training Quantization)

不需要重新训练，推理时动态量化。

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit量化
model = AutoModelForCausalLM.from_pretrained(
    "merged-dpo",
    load_in_8bit=True,
    device_map="auto"
)

# 4-bit量化 (更激进)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "merged-dpo",
    quantization_config=quantization_config,
    device_map="auto"
)
```

**性能对比** (7B模型):
| 精度 | 显存 | 速度 | 质量损失 |
|------|------|------|----------|
| BF16 | 14GB | 1.0x | 0% |
| INT8 | 7GB | 1.2x | <1% |
| INT4 | 3.5GB | 1.5x | 1-3% |

#### 2.1.2 量化感知训练 (QAT)

在训练时就考虑量化，质量损失最小。

```bash
# 使用已有的量化脚本
python model_quant.py \
    --model_name_or_path merged-dpo \
    --output_dir merged-dpo-int4 \
    --bits 4
```

---

### 2.2 KV Cache优化

**原理**: 缓存之前计算的key-value，避免重复计算。

**Multi-Query Attention (MQA)**:
- Key和Value在多个head间共享
- 显存占用降低

**Grouped-Query Attention (GQA)**:
- Key和Value在部分head间共享
- 平衡性能和质量

**代码示例**:
```python
# 在generate时自动启用
outputs = model.generate(
    **inputs,
    max_length=2048,
    use_cache=True,  # 启用KV cache
    do_sample=True
)
```

---

### 2.3 Flash Attention

**原理**: 重新组织attention计算，减少HBM访问，加速2-4倍。

**安装**:
```bash
pip install flash-attn --no-build-isolation
```

**使用**:
```python
model = AutoModelForCausalLM.from_pretrained(
    "merged-dpo",
    attn_implementation="flash_attention_2",  # 启用Flash Attention 2
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

**性能提升**:
- 速度: 2-4x
- 显存: 降低30-50%
- 支持更长上下文

---

### 2.4 投机采样 (Speculative Decoding)

**原理**: 用小模型预测多个token，大模型并行验证，加速2-3倍。

**代码示例**:
```python
from transformers import AutoModelForCausalLM

# 大模型
target_model = AutoModelForCausalLM.from_pretrained("merged-dpo")

# 小模型(draft model)
draft_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# 使用投机采样
outputs = target_model.generate(
    **inputs,
    assistant_model=draft_model,  # 指定draft model
    do_sample=False
)
```

**适用场景**:
- 长文本生成
- 对延迟敏感的场景
- 有合适的小模型可用

---

### 2.5 PagedAttention (vLLM核心技术)

**原理**: 将KV cache分页管理，类似操作系统的虚拟内存。

**优势**:
- KV cache利用率接近100%
- 支持更大的batch size
- 动态内存管理

**如何使用**: 见[3.2 vLLM部署](#32-vllm部署)

---

## 3. 部署方案

### 3.1 本地部署

#### 3.1.1 FastAPI服务

适合小规模部署，支持基本的负载均衡。

```bash
# 启动服务
python inference_api.py --model_path merged-dpo --port 8000

# 客户端调用
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "history": []}'
```

**特点**:
- ✅ 实现简单
- ✅ 易于定制
- ❌ 并发性能一般

#### 3.1.2 Gradio界面

快速搭建Web UI，适合演示和内部使用。

```bash
python inference_gradio.py --model_path merged-dpo --share
```

**特点**:
- ✅ 零前端代码
- ✅ 美观易用
- ✅ 支持分享链接

---

### 3.2 vLLM部署

**vLLM** 是当前最流行的高性能推理引擎。

**核心技术**:
- PagedAttention
- Continuous Batching
- 优化的CUDA kernels

**安装**:
```bash
pip install vllm
```

**启动服务**:
```bash
# 方式1: Python API
python -m vllm.entrypoints.openai.api_server \
    --model merged-dpo \
    --port 8000 \
    --dtype bfloat16

# 方式2: 使用脚本
bash vllm_deployment.sh
```

**客户端调用**:
```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="merged-dpo",
    messages=[
        {"role": "user", "content": "你好"}
    ]
)

print(response.choices[0].message.content)
```

**性能对比** (vs HuggingFace):
- 吞吐量: 10-20x ⬆️
- 延迟: 2-5x ⬇️
- 支持更大batch size

**vLLM高级功能**:
```bash
# 启用张量并行(多卡)
python -m vllm.entrypoints.openai.api_server \
    --model merged-dpo \
    --tensor-parallel-size 2  # 使用2张GPU

# 启用量化
python -m vllm.entrypoints.openai.api_server \
    --model merged-dpo \
    --quantization awq  # 或 gptq, squeezellm
```

---

### 3.3 TGI部署 (Text Generation Inference)

**HuggingFace官方推理引擎**，功能类似vLLM。

**Docker部署**:
```bash
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $(pwd)/merged-dpo:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id /data \
    --num-shard 1
```

**特点**:
- ✅ HuggingFace生态集成好
- ✅ 支持更多模型架构
- ❌ 性能略逊于vLLM

---

### 3.4 TensorRT-LLM部署

**NVIDIA官方方案**，性能最强。

**步骤**:
1. 模型转换为TensorRT格式
2. 优化和校准
3. 部署推理

**性能**:
- 吞吐量: 比vLLM再高30-50%
- 延迟: 最低

**缺点**:
- 部署复杂
- 模型转换耗时
- 调试困难

---

### 3.5 云端部署

#### 3.5.1 SageMaker (AWS)

```python
from sagemaker.huggingface import HuggingFaceModel

huggingface_model = HuggingFaceModel(
    model_data="s3://my-bucket/merged-dpo",
    role=role,
    transformers_version="4.49",
    pytorch_version="2.0",
    py_version="py310",
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge"
)
```

#### 3.5.2 Vertex AI (Google Cloud)

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

model = aiplatform.Model.upload(
    display_name="my-medical-gpt",
    artifact_uri="gs://my-bucket/merged-dpo",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu:latest"
)

endpoint = model.deploy(
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1
)
```

---

## 4. 性能对比

### 4.1 推理框架对比 (7B模型, A100 40GB)

| 框架 | 吞吐量 (tokens/s) | 延迟 (ms) | 显存占用 | 易用性 |
|------|-------------------|-----------|----------|--------|
| HuggingFace | 50 | 200 | 28GB | ⭐⭐⭐⭐⭐ |
| vLLM | 500-800 | 40-60 | 20GB | ⭐⭐⭐⭐ |
| TGI | 400-600 | 50-80 | 22GB | ⭐⭐⭐⭐ |
| TensorRT-LLM | 800-1200 | 30-40 | 18GB | ⭐⭐ |

### 4.2 量化对比 (7B模型)

| 方法 | 显存 | 速度提升 | MMLU准确率 | 推荐度 |
|------|------|----------|------------|--------|
| FP16 | 14GB | 1.0x | 65.2% | ⭐⭐⭐ |
| BF16 | 14GB | 1.0x | 65.3% | ⭐⭐⭐⭐⭐ |
| INT8 | 7GB | 1.2x | 64.8% | ⭐⭐⭐⭐ |
| INT4 (GPTQ) | 3.5GB | 1.5x | 63.5% | ⭐⭐⭐⭐ |
| INT4 (AWQ) | 3.5GB | 1.6x | 64.2% | ⭐⭐⭐⭐⭐ |

---

## 5. 推荐方案

### 场景1: 开发测试
**推荐**: HuggingFace + INT8量化
```bash
python inference.py --base_model merged-dpo --load_in_8bit
```

### 场景2: 生产部署 (中等流量)
**推荐**: vLLM + BF16
```bash
python -m vllm.entrypoints.openai.api_server \
    --model merged-dpo \
    --dtype bfloat16
```

### 场景3: 生产部署 (高流量)
**推荐**: vLLM + 多卡 + INT4
```bash
python -m vllm.entrypoints.openai.api_server \
    --model merged-dpo \
    --tensor-parallel-size 4 \
    --quantization awq
```

### 场景4: 极致性能
**推荐**: TensorRT-LLM + 多卡
- 吞吐量最高
- 延迟最低
- 部署复杂度高

### 场景5: 低成本部署
**推荐**: INT4量化 + CPU推理
```python
# 使用llama.cpp
pip install llama-cpp-python

from llama_cpp import Llama

llm = Llama(model_path="merged-dpo-q4.gguf")
output = llm("你好", max_tokens=100)
```

---

## 6. 下一步优化

想要进一步提升性能，可以尝试：

1. **模型蒸馏**: 训练更小的模型，保持性能
2. **混合精度**: 关键层用FP16，其他用INT8
3. **动态batch**: 根据负载自动调整batch size
4. **模型并行**: 超大模型跨卡部署
5. **Mixture of Experts**: 只激活部分参数

## 📚 参考资源

- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [HuggingFace TGI](https://github.com/huggingface/text-generation-inference)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [AWQ量化](https://github.com/mit-han-lab/llm-awq)
