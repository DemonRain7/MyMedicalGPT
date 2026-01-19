# -*- coding: utf-8 -*-
"""
vLLM 高性能推理服务
High-performance inference with vLLM

vLLM 特性:
- PagedAttention: 高效 KV cache 管理
- Continuous Batching: 动态批处理
- 10-20x 吞吐量提升

Usage:
    # 交互式推理
    python inference_vllm.py --model_path merged-dpo --interactive

    # 启动 OpenAI 兼容 API 服务
    python inference_vllm.py --model_path merged-dpo --serve --port 8000

    # 批量推理
    python inference_vllm.py --model_path merged-dpo --data_file input.jsonl --output_file output.jsonl

Requirements:
    pip install vllm
"""

import argparse
import json
import os
from typing import List, Optional

# 检查 vLLM 是否安装
try:
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.openai.api_server import run_server
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not installed. Install with: pip install vllm")


def create_llm(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    dtype: str = "auto",
    quantization: Optional[str] = None,
) -> "LLM":
    """
    创建 vLLM 引擎

    Args:
        model_path: 模型路径
        tensor_parallel_size: 张量并行数量（多卡时使用）
        gpu_memory_utilization: GPU 显存使用率
        max_model_len: 最大序列长度
        dtype: 数据类型 (auto, float16, bfloat16)
        quantization: 量化方法 (awq, gptq, squeezellm, None)
    """
    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM not installed. Install with: pip install vllm")

    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "dtype": dtype,
        "trust_remote_code": True,
    }

    if quantization:
        llm_kwargs["quantization"] = quantization

    return LLM(**llm_kwargs)


def generate_response(
    llm: "LLM",
    prompts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    stop: Optional[List[str]] = None,
) -> List[str]:
    """
    使用 vLLM 生成响应

    Args:
        llm: vLLM 引擎实例
        prompts: 输入提示列表
        max_tokens: 最大生成 token 数
        temperature: 温度参数
        top_p: Top-p 采样
        top_k: Top-k 采样
        repetition_penalty: 重复惩罚
        stop: 停止词列表

    Returns:
        生成的响应列表
    """
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stop=stop or [],
    )

    outputs = llm.generate(prompts, sampling_params)

    responses = []
    for output in outputs:
        generated_text = output.outputs[0].text
        responses.append(generated_text)

    return responses


def build_chat_prompt(
    message: str,
    history: List[List[str]] = None,
    system_prompt: str = None,
    template: str = "qwen"
) -> str:
    """
    构建对话格式的 prompt

    Args:
        message: 用户消息
        history: 对话历史 [[user1, assistant1], [user2, assistant2], ...]
        system_prompt: 系统提示
        template: 对话模板 (qwen, llama, chatglm)
    """
    history = history or []

    if template == "qwen":
        # Qwen chat template
        prompt = ""
        if system_prompt:
            prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

        for user_msg, assistant_msg in history:
            prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"

        prompt += f"<|im_start|>user\n{message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

    elif template == "llama":
        # Llama 2/3 chat template
        prompt = ""
        if system_prompt:
            prompt += f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"

        for i, (user_msg, assistant_msg) in enumerate(history):
            if i == 0 and system_prompt:
                prompt += f"{user_msg} [/INST] {assistant_msg} </s>"
            else:
                prompt += f"<s>[INST] {user_msg} [/INST] {assistant_msg} </s>"

        if history and system_prompt:
            prompt += f"<s>[INST] {message} [/INST]"
        elif system_prompt:
            prompt += f"{message} [/INST]"
        else:
            prompt += f"<s>[INST] {message} [/INST]"

    else:
        # Default simple template
        prompt = ""
        if system_prompt:
            prompt += f"System: {system_prompt}\n\n"

        for user_msg, assistant_msg in history:
            prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"

        prompt += f"User: {message}\nAssistant: "

    return prompt


def interactive_chat(
    llm: "LLM",
    system_prompt: str = None,
    template: str = "qwen",
    max_tokens: int = 512,
    temperature: float = 0.7,
):
    """
    交互式对话
    """
    print("=" * 50)
    print("vLLM 交互式对话")
    print("输入 'exit' 退出, 'clear' 清除历史")
    print("=" * 50)

    history = []

    # 获取停止词
    stop_tokens = ["<|im_end|>", "</s>", "[/INST]"]

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出...")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("退出...")
            break

        if user_input.lower() == "clear":
            history = []
            print("历史已清除")
            continue

        # 构建 prompt
        prompt = build_chat_prompt(
            user_input,
            history,
            system_prompt,
            template
        )

        # 生成响应
        responses = generate_response(
            llm,
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_tokens,
        )

        response = responses[0].strip()
        print(f"\nAssistant: {response}")

        # 更新历史
        history.append([user_input, response])


def batch_inference(
    llm: "LLM",
    data_file: str,
    output_file: str,
    system_prompt: str = None,
    template: str = "qwen",
    max_tokens: int = 512,
    temperature: float = 0.7,
    batch_size: int = 32,
):
    """
    批量推理

    输入文件格式 (jsonl):
        {"query": "问题1"}
        {"query": "问题2"}

    输出文件格式 (jsonl):
        {"query": "问题1", "response": "回答1"}
        {"query": "问题2", "response": "回答2"}
    """
    # 读取输入
    queries = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            queries.append(data.get("query", data.get("input", data.get("question", ""))))

    print(f"Loaded {len(queries)} queries from {data_file}")

    # 构建 prompts
    prompts = [
        build_chat_prompt(q, system_prompt=system_prompt, template=template)
        for q in queries
    ]

    # 停止词
    stop_tokens = ["<|im_end|>", "</s>", "[/INST]"]

    # 批量生成
    all_responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}")

        responses = generate_response(
            llm,
            batch_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_tokens,
        )
        all_responses.extend(responses)

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        for query, response in zip(queries, all_responses):
            result = {"query": query, "response": response.strip()}
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_responses)} responses to {output_file}")


def serve_openai_api(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    dtype: str = "auto",
    quantization: Optional[str] = None,
):
    """
    启动 OpenAI 兼容的 API 服务

    启动后可以使用 OpenAI Python SDK 调用:

        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        response = client.chat.completions.create(
            model="merged-dpo",
            messages=[{"role": "user", "content": "你好"}]
        )
    """
    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--dtype", dtype,
        "--trust-remote-code",
    ]

    if quantization:
        cmd.extend(["--quantization", quantization])

    print(f"Starting vLLM OpenAI API server...")
    print(f"Command: {' '.join(cmd)}")
    print(f"\nAPI will be available at: http://{host}:{port}/v1")
    print(f"\nExample usage:")
    print(f'  curl http://localhost:{port}/v1/chat/completions \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"model": "{model_path}", "messages": [{{"role": "user", "content": "你好"}}]}}\'')
    print()

    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="vLLM 高性能推理")

    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="张量并行数量（多卡）")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU 显存使用率")
    parser.add_argument("--max_model_len", type=int, default=4096, help="最大序列长度")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"], help="数据类型")
    parser.add_argument("--quantization", type=str, default=None, choices=["awq", "gptq", "squeezellm"], help="量化方法")

    # 生成参数
    parser.add_argument("--max_tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度")
    parser.add_argument("--template", type=str, default="qwen", choices=["qwen", "llama", "default"], help="对话模板")
    parser.add_argument("--system_prompt", type=str, default=None, help="系统提示")

    # 运行模式
    parser.add_argument("--interactive", action="store_true", help="交互式对话模式")
    parser.add_argument("--serve", action="store_true", help="启动 OpenAI 兼容 API 服务")
    parser.add_argument("--data_file", type=str, default=None, help="批量推理输入文件")
    parser.add_argument("--output_file", type=str, default="vllm_output.jsonl", help="批量推理输出文件")
    parser.add_argument("--batch_size", type=int, default=32, help="批量推理 batch size")

    # 服务参数
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")

    args = parser.parse_args()

    if not VLLM_AVAILABLE:
        print("Error: vLLM not installed.")
        print("Install with: pip install vllm")
        print("\nNote: vLLM requires Linux and NVIDIA GPU with CUDA support.")
        return

    # 启动 API 服务模式
    if args.serve:
        serve_openai_api(
            model_path=args.model_path,
            host=args.host,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            quantization=args.quantization,
        )
        return

    # 创建 LLM 引擎
    print(f"Loading model: {args.model_path}")
    llm = create_llm(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        quantization=args.quantization,
    )
    print("Model loaded!")

    # 交互模式
    if args.interactive:
        interactive_chat(
            llm,
            system_prompt=args.system_prompt,
            template=args.template,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    # 批量推理模式
    elif args.data_file:
        batch_inference(
            llm,
            data_file=args.data_file,
            output_file=args.output_file,
            system_prompt=args.system_prompt,
            template=args.template,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )

    else:
        # 默认：简单测试
        print("\nTesting with a simple query...")
        prompt = build_chat_prompt("你好，请介绍一下你自己", template=args.template)
        responses = generate_response(llm, [prompt], max_tokens=args.max_tokens)
        print(f"\nResponse: {responses[0]}")
        print("\nUse --interactive for chat mode, --serve for API server, or --data_file for batch inference.")


if __name__ == "__main__":
    main()
