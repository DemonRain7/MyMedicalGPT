# -*- coding: utf-8 -*-
"""
Gradio Web Interface for Model Inference
基于Gradio的Web界面

Usage:
    python inference_gradio.py --model_path merged-dpo
"""

import argparse
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger

class ChatBot:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model from {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    def chat(
        self,
        message: str,
        history: list,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """Generate response"""
        # Build prompt
        prompt = ""
        for user_msg, bot_msg in history:
            prompt += f"USER: {user_msg}\nASSISTANT: {bot_msg}\n"
        prompt += f"USER: {message}\nASSISTANT:"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("ASSISTANT:")[-1].strip()

        return response

def create_interface(bot: ChatBot):
    """Create Gradio interface"""

    def respond(message, chat_history, max_length, temperature, top_p):
        bot_message = bot.chat(message, chat_history, max_length, temperature, top_p)
        chat_history.append((message, bot_message))
        return "", chat_history

    with gr.Blocks(title="MyMedicalGPT Chat") as demo:
        gr.Markdown("# MyMedicalGPT Chat Interface")
        gr.Markdown("基于自训练模型的对话界面")

        chatbot = gr.Chatbot(height=400)

        with gr.Row():
            msg = gr.Textbox(
                placeholder="输入你的问题...",
                show_label=False,
                scale=4
            )
            submit = gr.Button("发送", scale=1)

        with gr.Accordion("高级设置", open=False):
            max_length = gr.Slider(
                minimum=512,
                maximum=4096,
                value=2048,
                step=128,
                label="最大长度"
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="温度"
            )
            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top P"
            )

        clear = gr.Button("清除对话")

        # Event handlers
        submit.click(
            respond,
            [msg, chatbot, max_length, temperature, top_p],
            [msg, chatbot]
        )

        msg.submit(
            respond,
            [msg, chatbot, max_length, temperature, top_p],
            [msg, chatbot]
        )

        clear.click(lambda: None, None, chatbot, queue=False)

    return demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run server")
    args = parser.parse_args()

    # Initialize bot
    bot = ChatBot(args.model_path)

    # Create and launch interface
    demo = create_interface(bot)
    demo.launch(share=args.share, server_port=args.port)

if __name__ == "__main__":
    main()
