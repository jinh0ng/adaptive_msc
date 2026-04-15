# test_llm.py
from llm import call_llm

result = call_llm([
    {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
])
print("=== 모델 응답 ===")
print(repr(result))   # repr로 숨은 특수문자 확인
print(result)