#!/usr/bin/env python3
"""
E4B 모델로 GSM8K 1문제 파이프라인 동작 확인
"""
import os, sys
sys.path.insert(0, "/data/yejinhong/adaptive_ttc")

import config
config.MODEL_PATH = "/data/yejinhong/gemma-4-E4B-it"

from benchmark.gsm8k import load_gsm8k, is_correct
from llm import call_llm
from prompts.solve import build_solve_prompt

print("=" * 60)
print(f"  [Smoke Test] E4B 모델 / GSM8K 1문제")
print(f"  MODEL_PATH : {config.MODEL_PATH}")
print("=" * 60)

dataset = load_gsm8k(split="test", n_samples=1)
item     = dataset[0]
question = item["question"]
gold     = item["answer"]

print(f"\n📝 문제:\n{question}\n")
print(f"✅ 정답 (gold):\n{gold}\n")
print("-" * 60)
print("[LLM] 추론 시작...")

pred    = call_llm(build_solve_prompt(question))
correct = is_correct(pred, gold)

print(f"\n🤖 모델 출력:\n{pred}\n")
print("-" * 60)
print(f"\n결과: {'✅ 정답!' if correct else '❌ 오답'}")
print("=" * 60)
