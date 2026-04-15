# prompts/decompose.py

def build_decompose_prompt(task: str, failed_answer: str, error: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a task decomposer. "
                "Break the math problem into 2-3 smaller, simpler sub-problems. "
                "Each sub-problem must be independently solvable and together they should lead to the final answer. "
                "Format your response as a numbered list:\n"
                "1. <sub-problem>\n"
                "2. <sub-problem>\n"
                "3. <sub-problem> (optional)"
            )
        },
        {
            "role": "user",
            "content": (
                f"Problem: {task}\n\n"
                f"Previous failed answer: {failed_answer}\n"
                f"Error identified: {error}\n\n"
                "Break this into simpler sub-problems:"
            )
        }
    ]