# agents/root_agent.py
import re
from agents.base_agent import BaseAgent
from prompts.solve import build_solve_prompt
from prompts.reflection import build_reflection_prompt
from prompts.decompose import build_decompose_prompt
from config import MAX_DEPTH, MAX_WIDTH

LEAF_ERROR_TAG = "[LEAF_ERROR]"
ORIGINAL_TAG   = "[Original problem for context]"
SUBTASK_TAG    = "[Your specific sub-task to solve]"

SKIP_PREFIXES = (
    "here is", "since", "please", "note that",
    "once you", "i cannot", "the following",
    "as you", "if you", "you can", "you need",
    "this problem", "the problem", "the user",
    "unfortunately", "however", "therefore",
)

class RootAgent(BaseAgent):
    def __init__(self, depth: int = 0, max_depth: int = MAX_DEPTH, max_width: int = MAX_WIDTH):
        super().__init__(depth=depth)
        self.max_depth = max_depth
        self.max_width = max_width
        self.is_leaf = (depth >= self.max_depth)

    def _extract_original(self, task: str) -> str:
        if SUBTASK_TAG in task:
            raw = task.split(SUBTASK_TAG)[0]
        else:
            raw = task
        return raw.replace(ORIGINAL_TAG, "").strip()

    def _is_no_error(self, reflection: str) -> bool:
        first_line = reflection.strip().split("\n")[0].strip()
        return first_line == "NO_ERROR" or reflection.strip().startswith("NO_ERROR")

    def _parse_leaf_error(self, error_text: str) -> tuple:
        reason, answer = "", ""
        if "[REASON]" in error_text and "[ANSWER]" in error_text:
            reason = error_text.split("[REASON]")[1].split("[ANSWER]")[0].strip()
            answer = error_text.split("[ANSWER]")[1].strip()
        else:
            answer = error_text.replace(LEAF_ERROR_TAG, "").strip()
        return reason, answer

    def run(self, task: str) -> str:
        print(f"\n{'  ' * self.depth}[Agent depth={self.depth} | leaf={self.is_leaf}] 문제 받음")
        print(f"{'  ' * self.depth}  task: {task[:80]}{'...' if len(task) > 80 else ''}")

        # Step 1: 풀기
        answer = self.call(build_solve_prompt(task))
        print(f"{'  ' * self.depth}  → 초안: {answer[:80]}{'...' if len(answer) > 80 else ''}")

        # Step 2: Self-reflection
        reflection = self.call(build_reflection_prompt(task, answer))
        print(f"{'  ' * self.depth}  → reflection: {reflection[:100]}{'...' if len(reflection) > 100 else ''}")

        # Step 3: 오류 없으면 반환
        if self._is_no_error(reflection):
            print(f"{'  ' * self.depth}  ✓ NO_ERROR → 확정")
            return answer

        # Step 4: leaf면 에러 태그 + reflection 이유 붙여서 반환
        if self.is_leaf:
            print(f"{'  ' * self.depth}  ✗ leaf 에러 → 상위 회신 (reason 포함)")
            return f"{LEAF_ERROR_TAG} [REASON] {reflection} [ANSWER] {answer}"

        # Step 5: 분해 — 원래 문제만 추출
        original = self._extract_original(task)
        print(f"{'  ' * self.depth}  → 분해 중...")
        decompose_result = self.call(build_decompose_prompt(original, answer, reflection))
        sub_tasks = self._parse_subtasks(decompose_result)
        print(f"{'  ' * self.depth}  → sub-task {len(sub_tasks)}개:")
        for i, st in enumerate(sub_tasks):
            print(f"{'  ' * self.depth}    {i+1}. {st[:60]}{'...' if len(st) > 60 else ''}")

        # Step 6: 하위 에이전트 실행
        sub_results = []
        for i, st in enumerate(sub_tasks[:self.max_width]):
            print(f"{'  ' * self.depth}  → sub-task {i+1} 하위 에이전트 실행")
            child = RootAgent(depth=self.depth + 1, max_depth=self.max_depth, max_width=self.max_width)
            child_task = (
                f"{ORIGINAL_TAG}\n{original}\n\n"
                f"{SUBTASK_TAG}\n{st}"
            )
            result = child.run(child_task)
            sub_results.append((st, result))

        # Step 7: merge
        final = self._merge(original, sub_results)
        print(f"{'  ' * self.depth}  ✓ merge 완료 → {final[:60]}{'...' if len(final) > 60 else ''}")
        return final

    def _parse_subtasks(self, text: str) -> list:
        lines = text.strip().split("\n")
        tasks = []
        for line in lines:
            cleaned = re.sub(r"^[\d]+[\.\)]\s*", "", line).strip()
            cleaned = re.sub(r"^[-\*]+\s*", "", cleaned).strip()
            cleaned = re.sub(r"^\*+\s*", "", cleaned).strip()
            if any(cleaned.lower().startswith(p) for p in SKIP_PREFIXES):
                continue
            if cleaned and len(cleaned) > 10:
                tasks.append(cleaned)
        return tasks if tasks else [text.strip()]

    def _merge(self, original_task: str, sub_results: list) -> str:
        valid   = [(st, ans) for st, ans in sub_results
                   if not ans.startswith(LEAF_ERROR_TAG)]
        invalid = [(st, ans) for st, ans in sub_results
                   if ans.startswith(LEAF_ERROR_TAG)]
        print(f"{'  ' * self.depth}  → merge: 유효 {len(valid)}개 / 에러 {len(invalid)}개")

        # 에러에서 reason 추출 → 힌트로 활용
        error_hints = []
        for st, err in invalid:
            reason, ans = self._parse_leaf_error(err)
            hint = f"Sub-problem: {st}\nFailed answer: {ans}"
            if reason:
                hint += f"\nError reason: {reason}"
            error_hints.append(hint)
        error_text = "\n\n".join(error_hints) if error_hints else ""

        # 유효 없으면 에러 힌트 기반 직접 풀기
        if not valid:
            print(f"{'  ' * self.depth}  → 유효 없음, 에러 힌트로 직접 풀기")
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a math solver. "
                        "Some sub-problems failed. "
                        "Use the error reasons as hints to avoid the same mistakes. "
                        "End your response with exactly: ANSWER: <number>"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Main problem: {original_task}\n\n"
                        f"Failed attempts and their error reasons:\n{error_text}\n\n"
                        "Now solve the problem correctly, avoiding the errors above."
                    )
                }
            ]
            return self.call(messages)

        # 유효한 결과 + 에러 힌트 함께 merge
        valid_text = "\n".join(
            [f"Sub-problem {i+1}: {st}\nSub-answer {i+1}: {ans}"
             for i, (st, ans) in enumerate(valid)]
        )
        user_content = (
            f"Main problem: {original_task}\n\n"
            f"Successful sub-answers:\n{valid_text}"
        )
        if error_text:
            user_content += (
                f"\n\nFailed sub-attempts (use as hints to avoid mistakes):\n{error_text}"
            )
        user_content += "\n\nNow give the final answer to the main problem."

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a math solver. "
                    "You are given a main problem, successful sub-answers, "
                    "and failed attempts with error reasons. "
                    "Use the successful answers and learn from the errors "
                    "to give the correct final answer. "
                    "End your response with exactly: ANSWER: <number>"
                )
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        return self.call(messages)