# test_agent_force_decompose.py
# 분해 흐름 강제 테스트 — reflection과 decompose만 직접 호출
from llm import call_llm
from prompts.reflection import build_reflection_prompt
from prompts.decompose import build_decompose_prompt
from agents.root_agent import RootAgent

task = (
    "A train travels 60 miles per hour for 2 hours, "
    "then 80 miles per hour for 3 hours. "
    "What is the total distance traveled?"
)
wrong_answer = "ANSWER: 200"  # 일부러 틀린 답 (정답은 280)

print("="*60)
print(f"문제: {task}")
print(f"주입할 틀린 답: {wrong_answer}")
print("="*60)

# reflection 테스트
print("\n--- reflection 테스트 ---")
reflection = call_llm(build_reflection_prompt(task, wrong_answer))
print(f"reflection 결과: {reflection}")

# decompose 테스트
print("\n--- decompose 테스트 ---")
decompose = call_llm(build_decompose_prompt(task, wrong_answer, reflection))
print(f"decompose 결과:\n{decompose}")

# 실제 에이전트 전체 흐름 (이 문제는 어렵지 않아서 NO_ERROR 날 수도 있음)
print("\n--- 실제 에이전트 전체 흐름 ---")
agent = RootAgent(depth=0)
result = agent.run(task)
print(f"\n[최종 출력]\n{result}")