# -*- coding: utf-8 -*-
"""
Simple FastAPI Server for Model Inference
基于FastAPI的简单推理服务

Usage:
    python inference_api.py --model_path merged-dpo --port 8000

API Endpoints:
    POST /chat - Single query
    POST /chat/stream - Streaming response
"""

import argparse
import torch
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import uvicorn
import json

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[List[str]]] = []
    max_length: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class ChatResponse(BaseModel):
    response: str
    history: List[List[str]]

# Initialize FastAPI app
app = FastAPI(title="MyMedicalGPT API", version="1.0")

# Global variables for model
tokenizer = None
model = None
device = None

def load_model(model_path: str):
    """Load model and tokenizer"""
    global tokenizer, model, device

    logger.info(f"Loading model from {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model.eval()

    logger.info(f"Model loaded on {device}")

def generate_response(
    message: str,
    history: List[List[str]] = [],
    max_length: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """Generate response from model"""
    # Build prompt with history
    prompt = ""
    for user_msg, assistant_msg in history:
        prompt += f"USER: {user_msg}\nASSISTANT: {assistant_msg}\n"
    prompt += f"USER: {message}\nASSISTANT:"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant's response
    response = response.split("ASSISTANT:")[-1].strip()

    return response

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Single turn chat endpoint"""
    try:
        response = generate_response(
            request.message,
            request.history,
            request.max_length,
            request.temperature,
            request.top_p
        )

        # Update history
        new_history = request.history + [[request.message, response]]

        return ChatResponse(response=response, history=new_history)

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint"""
    async def generate():
        try:
            response = generate_response(
                request.message,
                request.history,
                request.max_length,
                request.temperature,
                request.top_p
            )

            # Simulate streaming
            for i in range(0, len(response), 5):
                chunk = response[i:i+5]
                yield f"data: {json.dumps({'text': chunk})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "model_loaded": model is not None}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    args = parser.parse_args()

    # Load model
    load_model(args.model_path)

    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
