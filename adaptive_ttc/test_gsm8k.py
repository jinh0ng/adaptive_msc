# test_gsm8k.py
from benchmark.gsm8k import load_gsm8k, extract_gold_answer, extract_pred_answer, is_correct

# 1. 데이터 로딩 확인
print("=== 데이터 로딩 확인 ===")
test_data = load_gsm8k(split="test", n_samples=3)
train_data = load_gsm8k(split="train", n_samples=3)
print(f"test 샘플 수: {len(test_data)}")
print(f"train 샘플 수: {len(train_data)}")

print("\n=== test 첫 번째 샘플 ===")
sample = test_data[0]
print(f"question: {sample['question']}")
print(f"answer: {sample['answer']}")

# 2. 정답 파싱 확인
print("\n=== 정답 파싱 확인 ===")
gold = extract_gold_answer(sample["answer"])
print(f"추출된 정답: {gold}")

# 3. 채점 확인
print("\n=== 채점 확인 ===")
cases = [
    ("ANSWER: " + str(int(gold)), sample["answer"], True),   # 정답
    ("ANSWER: 9999",              sample["answer"], False),   # 오답
    ("the answer is " + str(int(gold)), sample["answer"], True),  # 다른 형식
]
for pred, ans, expected in cases:
    result = is_correct(pred, ans)
    status = "✅" if result == expected else "❌"
    print(f"{status} pred='{pred}' → is_correct={result} (expected={expected})")