# test_datasets.py — 3개 데이터셋 샘플 확인
from benchmark.aime import load_aime, is_correct as aime_correct
from benchmark.mmmlu import load_mmmlu, format_question, is_correct as mmmlu_correct
from benchmark.gsm8k import load_gsm8k, is_correct as gsm_correct

print("="*60)
print("GSM8K 샘플")
print("="*60)
for i, item in enumerate(load_gsm8k(split="test", n_samples=2)):
    print(f"\n[{i+1}] Q: {item['question'][:80]}...")
    print(f"     A: {item['answer'][-30:]}")

print()
print("="*60)
print("AIME 2026 샘플")
print("="*60)
for i, item in enumerate(load_aime(n_samples=2)):
    print(f"\n[{i+1}] Q: {item['problem'][:80]}...")
    print(f"     A: {item['answer']}")

print()
print("="*60)
print("MMMLU (KO-KR) 샘플")
print("="*60)
from benchmark.mmmlu import format_question
for i, item in enumerate(load_mmmlu(lang="ko", n_samples=2)):
    print(f"\n[{i+1}] Q: {format_question(item)[:120]}...")
    print(f"     A: {item['answer']} (Subject: {item['subject']})")

print()
print("="*60)
print("채점 함수 테스트")
print("="*60)
gsm_sample = load_gsm8k(split="test", n_samples=1)[0]
aime_sample = load_aime(n_samples=1)[0]
mmmlu_sample = load_mmmlu(lang="ko", n_samples=1)[0]

print(f"GSM8K  정답맞춤: {gsm_correct('ANSWER: 18', gsm_sample['answer'])}")
print(f"GSM8K  오답:     {gsm_correct('ANSWER: 999', gsm_sample['answer'])}")
print(f"AIME   정답맞춤: {aime_correct('ANSWER: 277', aime_sample['answer'])}")
print(f"AIME   오답:     {aime_correct('ANSWER: 999', aime_sample['answer'])}")
print(f"MMMLU  정답맞춤: {mmmlu_correct('ANSWER: B', mmmlu_sample['answer'])}")
print(f"MMMLU  오답:     {mmmlu_correct('ANSWER: A', mmmlu_sample['answer'])}")