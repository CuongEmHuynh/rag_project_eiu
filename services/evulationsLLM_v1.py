import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = [
    # Vietnamese-focused
    "phamhai/Llama-3.2-3B-Instruct-Frog",
    "vilm/vinallama-2.7b-chat",
    "arcee-ai/Arcee-VyLinh",
    "AITeamVN/Vi-Qwen2-3B-RAG",
    "AITeamVN/Vi-Qwen2-1.5B-RAG",
    "ricepaper/vi-gemma-2b-RAG",
    "thangvip/vilord-1.8B-instruct",
    # Multilingual
    "Qwen/Qwen2.5-1.5B-Instruct",
    "facebook/xglm-2.9B",
    "mistralai/Ministral-3-3B-Instruct-2512-BF16",
]

def check_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = check_device()



def build_context(scored_points, top_k=4):
    blocks = []
    for i, sp in enumerate(scored_points[:top_k], start=1):
        p = sp["payload"] if isinstance(sp, dict) else sp.payload
        blocks.append(
            f"[E{i}] doc_no: {p.get('doc_no')}\n"
            f"date: {p.get('date')}\n"
            f"summary: {p.get('summary')}\n"
            f"chunk_index: {p.get('chunk_index')}\n"
            f"file_url: {p.get('file_url')}\n"
            f"text:\n{p.get('chunk_text')}\n"
        )
    return "\n---\n".join(blocks)


def make_prompt(query, context):
    return f"""Bạn là trợ lý QA cho văn bản hành chính tiếng Việt (nguồn OCR).
CHỈ được dùng thông tin trong CONTEXT. Không bịa.
Khi trả lời, phải trích dẫn bằng [E1], [E2]... đúng evidence.

CONTEXT:
{context}

QUESTION:
{query}

YÊU CẦU TRẢ LỜI:
1) Trả lời ngắn gọn, đúng trọng tâm.
2) Nếu hỏi về điều khoản, hãy trích nguyên ý theo OCR và nêu "Điều x".
3) Luôn kèm trích dẫn [E?] sau mỗi ý quan trọng.
4) Nếu CONTEXT không đủ để trả lời, nói rõ "Không đủ dữ liệu trong context".
"""

def load_model(model_id):
    dtype = torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    if DEVICE == "mps":
        model = model.to("mps")
    elif DEVICE == "cpu":
        model = model.to("cpu")

    model.eval()
    return tok, model

def chat_generate(tok, model, prompt, max_new_tokens=256):
    # Lý do: một số model có chat template, dùng sẽ “đúng format” hơn
    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "Bạn là trợ lý QA tiếng Việt."},
            {"role": "user", "content": prompt},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Lý do: do_sample=False giúp giảm lặp/hallucination khi RAG
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.1,
            eos_token_id=tok.eos_token_id,
        )

    decoded = tok.decode(out[0], skip_special_tokens=True)
    return decoded

def main():
    pass


if __name__ == "__main__":
    main()

    #