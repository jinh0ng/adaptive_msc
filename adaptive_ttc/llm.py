# llm.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_PATH, MAX_NEW_TOKENS

_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    print(f"[LLM] 모델 로딩 중: {MODEL_PATH}")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    print("[LLM] 로딩 완료!")
    return _model, _tokenizer

def call_llm(messages: list[dict]) -> str:
    model, tokenizer = load_model()

    # chat template 적용
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 입력 제거하고 새로 생성된 부분만
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return result.strip()