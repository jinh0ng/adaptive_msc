# test_agent_full_flow.py
# 목적: decompose → 하위 에이전트 실행 → merge 전체 흐름 강제 테스트

from agents.base_agent import BaseAgent
from agents.root_agent import RootAgent
from prompts.solve import build_solve_prompt
from prompts.reflection import build_reflection_prompt
from prompts.decompose import build_decompose_prompt
from config import MAX_DEPTH, MAX_WIDTH
import re

# RootAgent를 상속해서 _solve만 오버라이드
# → 첫 번째 시도에서 일부러 틀린 답을 반환하게 패치
class PatchedRootAgent(RootAgent):
    def __init__(self, depth: int = 0, force_wrong: bool = False):
        super().__init__(depth=depth)
        self.force_wrong = force_wrong  # True면 첫 풀이를 틀리게 반환

    def run(self, task: str) -> str:
        print(f"\n{'  ' * self.depth}[Agent depth={self.depth}] 문제 받음")
        print(f"{'  ' * self.depth}  task: {task[:80]}{'...' if len(task) > 80 else ''}")

        # Step 1: 풀기 (force_wrong이면 틀린 답 주입)
        if self.force_wrong:
            answer = "ANSWER: 999"  # 의도적으로 틀린 답
            print(f"{'  ' * self.depth}  → [패치됨] 틀린 답 주입: {answer}")
        else:
            answer = self.call(build_solve_prompt(task))
            print(f"{'  ' * self.depth}  → 초안 답변: {answer[:80]}{'...' if len(answer) > 80 else ''}")

        # Step 2: Self-reflection
        reflection = self.call(build_reflection_prompt(task, answer))
        print(f"{'  ' * self.depth}  → reflection: {reflection[:100]}{'...' if len(reflection) > 100 else ''}")

        # Step 3: 오류 없으면 반환
        if "NO_ERROR" in reflection:
            print(f"{'  ' * self.depth}  ✓ NO_ERROR → 답 확정")
            return answer

        # Step 4: leaf면 그대로 반환
        if self.is_leaf:
            print(f"{'  ' * self.depth}  ✗ leaf → 상위로 반환")
            return answer

        # Step 5: 분해
        print(f"{'  ' * self.depth}  → 오류 발견, sub-task 분해 중...")
        decompose_result = self.call(build_decompose_prompt(task, answer, reflection))
        sub_tasks = self._parse_subtasks(decompose_result)
        print(f"{'  ' * self.depth}  → sub-task {len(sub_tasks)}개 생성:")
        for i, st in enumerate(sub_tasks):
            print(f"{'  ' * self.depth}    {i+1}. {st}")

        # Step 6: 하위 에이전트 실행 (하위는 force_wrong=False → 정상 풀이)
        sub_results = []
        for i, st in enumerate(sub_tasks[:MAX_WIDTH]):
            print(f"{'  ' * self.depth}  → sub-task {i+1} 하위 에이전트 실행")
            child = PatchedRootAgent(depth=self.depth + 1, force_wrong=False)
            result = child.run(st)
            sub_results.append((st, result))

        # Step 7: merge
        print(f"{'  ' * self.depth}  → 하위 결과 merge 중...")
        final = self._merge(task, sub_results)
        print(f"{'  ' * self.depth}  ✓ merge 완료")
        return final


# ── 테스트 실행 ──────────────────────────────────────────
task = (
    "A train travels 60 miles per hour for 2 hours, "
    "then 80 miles per hour for 3 hours. "
    "What is the total distance traveled?"
)

print("="*60)
print("전체 흐름 강제 테스트")
print(f"문제: {task}")
print(f"정답: 360 miles")
print(f"depth=0 에이전트: 틀린 답(999) 주입 → 분해 유도")
print(f"depth=1 에이전트: 정상 풀이")
print("="*60)

agent = PatchedRootAgent(depth=0, force_wrong=True)
result = agent.run(task)

print(f"\n{'='*60}")
print(f"[최종 출력]\n{result}")
print("="*60)