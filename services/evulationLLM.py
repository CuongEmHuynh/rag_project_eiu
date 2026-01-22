import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time
import torch
import json
CONFIG = "configs/models.yaml"
PROMPT_PATH ="prompt/test_prompt.txt"
OUT_JSONL = "results.jsonl"


def load_prompt_template(template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    return template

def run_model(hf_id, hits,max_new_tokens=256):
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(hf_id, device_map="cpu", torch_dtype="auto")

    prompt_tpl = load_prompt_template(PROMPT_PATH)

    system_msg = (
    "Bạn là trợ lý AI. Chỉ được dùng thông tin trong CONTEXT. "
    "Nếu không đủ thông tin, trả lời đúng: KHÔNG ĐỦ THÔNG TIN. "
    "Không lặp lại cùng một câu."
)
    
    messages = [
    {"role": "system", "content": system_msg},
    {"role": "user", "content": prompt_tpl},
]
    chat_text = tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

    inputs = tok(chat_text, return_tensors="pt", truncation=True, max_length=2048).to("cpu")
    input_len = inputs["input_ids"].shape[-1]
    

    gen_kwargs = dict(
    max_new_tokens=220,          # hạn chế lan man
    do_sample=False,             # deterministic để đánh giá
    temperature=0.0,
    repetition_penalty=1.15,     # chống lặp
    no_repeat_ngram_size=6,      # chặn lặp n-gram
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
)
    print(f"Running model: {hf_id}, input length: {input_len}")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
           **gen_kwargs,   # tránh warning + ổn định hơn
        )
   
    dt = time.time() - t0

    gen_ids = out[0][input_len:]
    answer = tok.decode(gen_ids, skip_special_tokens=True).strip()
    full_text = tok.decode(out[0], skip_special_tokens=True)


    dt = time.time() - t0

    row = {
        "model": hf_id,
        "prompt_path": PROMPT_PATH,
        "input_tokens": int(input_len),
        "generated_tokens": int(gen_ids.shape[-1]),
        "latency_sec": round(dt, 4),
        "answer": answer,
        "gen_kwargs": gen_kwargs,
}

    with open(OUT_JSONL, "w", encoding="utf-8") as w:
        w.write(json.dumps(row, ensure_ascii=False) + "\n")
    # lấy phần trả lời mới nhất: cắt theo prompt
    answer = full_text[len(prompt_tpl):].strip() if full_text.startswith(prompt_tpl) else full_text.strip()
    return answer,dt

def downloadModel():
    cfg = yaml.safe_load(open(CONFIG, "r", encoding="utf-8"))
    for m in cfg["models"]:
        hf_id = m["hf_id"]
        print(f"Downloading: {hf_id}")
        tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
        _ = AutoModelForCausalLM.from_pretrained(
            hf_id,
            device_map="cpu",
            torch_dtype="auto"
        )
        print(f"OK: {hf_id}")

def main():
    cfg = yaml.safe_load(open(CONFIG, "r", encoding="utf-8"))
    with open(OUT_JSONL, "w", encoding="utf-8") as w:
        for model_id in tqdm(cfg["models"], desc="Running models"):
            model_id = model_id["hf_id"]
            prompt = load_prompt_template(PROMPT_PATH)
            answer, latency = run_model(model_id, prompt, max_new_tokens=256)

            print(f"Model: {model_id}")
            print(f"Answer: {answer}")
            print(f"Latency: {latency:.2f} sec")
if __name__=="__main__":
    main()
    # downloadModel()