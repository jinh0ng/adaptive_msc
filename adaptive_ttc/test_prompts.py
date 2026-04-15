# test_prompts.py
from prompts.solve import build_solve_prompt
from prompts.reflection import build_reflection_prompt
from prompts.decompose import build_decompose_prompt

task = "Janet has 3 apples. She buys 5 more. How many does she have?"

print("=== solve prompt ===")
for m in build_solve_prompt(task):
    print(f"[{m['role']}] {m['content']}\n")

print("=== reflection prompt ===")
for m in build_reflection_prompt(task, "8"):
    print(f"[{m['role']}] {m['content']}\n")

print("=== decompose prompt ===")
for m in build_decompose_prompt(task, "7", "ERROR: addition mistake"):
    print(f"[{m['role']}] {m['content']}\n")