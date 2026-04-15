# prompts/solve_mcq.py — 객관식 전용

def build_solve_mcq_prompt(question_text: str) -> list[dict]:
    """MMMLU 객관식용 프롬프트 — 보기가 이미 question_text에 포함돼 있음"""
    return [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable assistant solving multiple choice questions. "
                "Analyze the question and the four options carefully. "
                "End your response with exactly: ANSWER: <A or B or C or D>"
            )
        },
        {
            "role": "user",
            "content": f"Question:\n{question_text}"
        }
    ]

def build_reflection_mcq_prompt(question_text: str, answer: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a multiple choice answer checker. "
                "Review the question and the chosen answer carefully. "
                "If the answer is correct, respond with exactly: NO_ERROR\n"
                "If there is a mistake, respond with exactly: ERROR: <brief description>"
            )
        },
        {
            "role": "user",
            "content": f"Question:\n{question_text}\n\nAnswer given: {answer}"
        }
    ]

def build_decompose_mcq_prompt(question_text: str, failed_answer: str, error: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a task decomposer for multiple choice questions. "
                "Break down the reasoning into 2-3 simpler steps to arrive at the correct answer. "
                "Format as a numbered list:\n"
                "1. <step>\n2. <step>\n3. <step> (optional)"
            )
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question_text}\n\n"
                f"Previous wrong answer: {failed_answer}\n"
                f"Error: {error}\n\n"
                "Break into simpler reasoning steps:"
            )
        }
    ]