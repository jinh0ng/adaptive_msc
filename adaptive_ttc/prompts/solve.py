# prompts/solve.py

def build_solve_prompt(task: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a precise math solver. "
                "Solve the problem step by step, then give the final answer. "
                "End your response with exactly: ANSWER: <number>"
            )
        },
        {
            "role": "user",
            "content": f"Problem: {task}"
        }
    ]