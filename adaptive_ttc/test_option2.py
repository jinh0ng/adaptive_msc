# test_option2.py
# Option 2 로직 검증: 하위 에이전트 1개 정상 + 1개 에러 시
# 상위가 정상 결과만으로 올바른 최종 답을 내는지 확인

from agents.root_agent import RootAgent, LEAF_ERROR_TAG

# ── 시나리오 설정 ─────────────────────────────────────
# sub-agent 1: 정상 (120 miles)
# sub-agent 2: 에러 ([LEAF_ERROR] 태그)
# 상위가 120만 보고 merge → 최종 답이 올바른지 확인

class MockChildAgent(RootAgent):
    """실제 LLM 호출 없이 미리 정해진 결과를 반환하는 mock 에이전트"""
    def __init__(self, depth: int, mock_result: str):
        super().__init__(depth=depth)
        self.mock_result = mock_result

    def run(self, task: str) -> str:
        print(f"\n{'  ' * self.depth}[MockAgent depth={self.depth}] 결과 반환: {self.mock_result[:60]}")
        return self.mock_result


class PatchedAgent(RootAgent):
    """
    depth=0 에이전트: 첫 풀이를 틀리게 → 분해 유도
    하위 에이전트: sub-task 1은 정상, sub-task 2는 [LEAF_ERROR]
    """
    def __init__(self, depth: int = 0):
        super().__init__(depth=depth)

    def run(self, task: str) -> str:
        print(f"\n{'  ' * self.depth}[PatchedAgent depth={self.depth}] 문제 받음")
        print(f"{'  ' * self.depth}  task: {task[:80]}")

        # depth=0: 틀린 답 주입 → 분해 유도
        if self.depth == 0:
            from prompts.reflection import build_reflection_prompt
            from prompts.decompose import build_decompose_prompt

            answer = "ANSWER: 999"
            print(f"{'  ' * self.depth}  → [패치] 틀린 답 주입: {answer}")

            reflection = self.call(build_reflection_prompt(task, answer))
            print(f"{'  ' * self.depth}  → reflection: {reflection[:100]}")

            if "NO_ERROR" in reflection:
                return answer

            print(f"{'  ' * self.depth}  → 분해 중...")
            decompose_result = self.call(build_decompose_prompt(task, answer, reflection))
            sub_tasks = self._parse_subtasks(decompose_result)
            print(f"{'  ' * self.depth}  → sub-task {len(sub_tasks)}개 생성")
            for i, st in enumerate(sub_tasks):
                print(f"{'  ' * self.depth}    {i+1}. {st}")

            # mock 하위 에이전트 주입
            # sub-task 1: 정상 결과
            # sub-task 2: LEAF_ERROR
            # sub-task 3 이상: 정상 결과
            sub_results = []
            for i, st in enumerate(sub_tasks[:3]):
                if i == 1:
                    mock = MockChildAgent(depth=1, mock_result=f"{LEAF_ERROR_TAG} ANSWER: 999")
                else:
                    child = RootAgent(depth=1)
                    mock = child  # 실제 LLM 호출 (정상 풀이)
                result = mock.run(st)
                sub_results.append((st, result))

            return self._merge(task, sub_results)

        # depth>=1: 정상 풀이
        return super().run(task)


# ── 테스트 실행 ───────────────────────────────────────
task = (
    "A train travels 60 miles per hour for 2 hours, "
    "then 80 miles per hour for 3 hours. "
    "What is the total distance traveled?"
)

print("="*60)
print("Option 2 검증 테스트")
print(f"문제: {task}")
print(f"정답: 360 miles")
print(f"시나리오: sub-task 2개 중 1개는 [LEAF_ERROR] 주입")
print("="*60)

agent = PatchedAgent(depth=0)
result = agent.run(task)

print(f"\n{'='*60}")
print(f"[최종 출력]\n{result}")

# 채점
from benchmark.gsm8k import extract_pred_answer
pred = extract_pred_answer(result)
print(f"\n추출된 숫자: {pred}")
print(f"정답(360)과 일치: {'✅' if pred == 360.0 else '❌'}")
print("="*60)