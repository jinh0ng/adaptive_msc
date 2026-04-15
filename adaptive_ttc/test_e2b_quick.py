#!/usr/bin/env python3
"""
E2B 모델로 GSM8K 1문제만 뽑아서 전체 파이프라인 동작 확인
  - llm.py / config.py 를 그대로 사용하되 MODEL_PATH만 E2B로 오버라이드
  - call_llm → is_correct 까지 확인
"""
import os, sys
sys.path.insert(0, "/data/yejinhong/adaptive_ttc")

# ── E2B로 오버라이드 ──────────────────────────────────────
os.environ["OVERRIDE_MODEL_PATH"] = "/data/yejinhong/gemma-4-E2B-it"

# config 를 임포트하기 전에 패치
import config
config.MODEL_PATH = "/data/yejinhong/gemma-4-E2B-it"

# ── 나머지 임포트 ─────────────────────────────────────────
from benchmark.gsm8k import load_gsm8k, is_correct
from llm import call_llm
from prompts.solve import build_solve_prompt

print("=" * 60)
print(f"  [Smoke Test] E2B 모델 / GSM8K 1문제")
print(f"  MODEL_PATH : {config.MODEL_PATH}")
print("=" * 60)

# 데이터 로드
dataset = load_gsm8k(split="test", n_samples=1)
item = dataset[0]
question = item["question"]
gold     = item["answer"]

print(f"\n📝 문제:\n{question}\n")
print(f"✅ 정답 (gold):\n{gold}\n")
print("-" * 60)
print("[LLM] 추론 시작... (시간 걸릴 수 있음)")

# 추론
pred = call_llm(build_solve_prompt(question))

print(f"\n🤖 모델 출력:\n{pred}\n")
print("-" * 60)

correct = is_correct(pred, gold)
print(f"\n결과: {'✅ 정답!' if correct else '❌ 오답'}")
print("=" * 60)
