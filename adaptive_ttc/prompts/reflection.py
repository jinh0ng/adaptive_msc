# prompts/reflection.py

def build_reflection_prompt(task: str, answer: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a math answer checker. "
                "Review the problem and the given answer carefully. "
                "If the answer is correct, respond with exactly: NO_ERROR\n"
                "If there is a mistake, respond with exactly: ERROR: <brief description of the mistake>"
            )
        },
        {
            "role": "user",
            "content": f"Problem: {task}\n\nAnswer given: {answer}"
        }
    ]