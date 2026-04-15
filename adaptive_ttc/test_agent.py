# test_agent.py
from agents.root_agent import RootAgent

# GSM8K 스타일 문제 2개로 테스트
problems = [
    {
        "question": "Janet has 3 apples. She buys 5 more and then gives 2 to her friend. How many apples does she have now?",
        "answer": "6"
    },
    {
        "question": "A store has 24 cookies. They sell 1/3 of them in the morning and 8 more in the afternoon. How many cookies are left?",
        "answer": "8"
    },
]

for i, prob in enumerate(problems):
    print(f"\n{'='*60}")
    print(f"문제 {i+1}: {prob['question']}")
    print(f"정답: {prob['answer']}")
    print(f"{'='*60}")

    agent = RootAgent(depth=0)
    result = agent.run(prob["question"])

    print(f"\n[최종 출력]\n{result}")
    print(f"{'='*60}")