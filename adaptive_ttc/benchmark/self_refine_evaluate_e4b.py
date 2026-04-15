# benchmark/self_refine_evaluate_e4b.py
import json
import csv
import os
from datetime import datetime
from tqdm import tqdm

from llm import call_llm
from benchmark.aime import load_aime, is_correct as aime_correct
from benchmark.mmmlu import load_mmmlu, format_question, is_correct as mmmlu_correct
from prompts.solve import build_solve_prompt 
from prompts.solve_mcq import build_solve_mcq_prompt
from prompts.reflection import build_reflection_prompt
from prompts.reflection import build_reflection_prompt as build_reflection_mcq
from config import MODEL_PATH

# Self-Refine 최대 반복 횟수 (원본 논문은 4회)
SELF_REFINE_MAX_ITER = 4


# ── Self-Refine 핵심 로직 ─────────────────────────────────

def build_refine_prompt(task: str, answer: str, feedback: str) -> list[dict]:
    """
    Self-Refine의 task_iterate에 해당
    피드백을 받아서 답을 개선하는 프롬프트
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a precise math solver. "
                "You are given a problem, your previous answer, and feedback on what went wrong. "
                "Use the feedback to correct your answer. "
                "End your response with exactly: ANSWER: <number>"
            )
        },
        {
            "role": "user",
            "content": (
                f"Problem: {task}\n\n"
                f"Your previous answer:\n{answer}\n\n"
                f"Feedback on your answer:\n{feedback}\n\n"
                "Now provide the corrected answer."
            )
        }
    ]

def build_refine_mcq_prompt(task: str, answer: str, feedback: str) -> list[dict]:
    """MMMLU 객관식용 refine 프롬프트"""
    return [
        {
            "role": "system",
            "content": (
                "You are a multiple choice solver. "
                "You are given a question, your previous answer, and feedback on what went wrong. "
                "Use the feedback to select the correct answer. "
                "End your response with exactly: ANSWER: <A or B or C or D>"
            )
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{task}\n\n"
                f"Your previous answer:\n{answer}\n\n"
                f"Feedback:\n{feedback}\n\n"
                "Now provide the corrected answer."
            )
        }
    ]

def _is_no_error(reflection: str) -> bool:
    """reflection 첫 줄 기준 NO_ERROR 판단"""
    first_line = reflection.strip().split("\n")[0].strip()
    return first_line == "NO_ERROR" or reflection.strip().startswith("NO_ERROR")

def self_refine(task: str, max_iter: int = SELF_REFINE_MAX_ITER,
                is_mcq: bool = False) -> tuple[str, list[dict]]:
    """
    Self-Refine 구현
    
    n=0: build_solve_prompt → 초안
    n=1~: build_reflection_prompt → feedback
          NO_ERROR면 break
          아니면 build_refine_prompt → 개선된 답
    """
    solve_fn   = build_solve_mcq_prompt if is_mcq else build_solve_prompt
    refine_fn  = build_refine_mcq_prompt if is_mcq else build_refine_prompt

    # Step 0: 초안 생성
    answer = call_llm(solve_fn(task))

    logs = [{"iter": 0, "answer": answer, "feedback": None, "stopped": False}]

    for i in range(1, max_iter + 1):
        # Step 1: Feedback 생성
        feedback = call_llm(build_reflection_prompt(task, answer))

        # 정지 조건
        if _is_no_error(feedback):
            logs.append({"iter": i, "answer": answer,
                         "feedback": feedback, "stopped": True})
            break

        # Step 2: Refine
        answer = call_llm(refine_fn(task, answer, feedback))
        logs.append({"iter": i, "answer": answer,
                     "feedback": feedback, "stopped": False})

    return answer, logs


# ── 유틸 ─────────────────────────────────────────────────

def score(results: list[dict]) -> dict:
    if not results:
        return {"accuracy_pct": 0.0, "correct": 0, "total": 0}
    n       = len(results)
    correct = sum(r["correct"] for r in results)
    return {
        "accuracy_pct": round(correct / n * 100, 2),
        "correct":      correct,
        "total":        n,
    }

def save_csv(rows: list[dict], path: str):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def save_json(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def print_progress(idx: int, total: int, question: str, gold,
                   sr_pred: str, sr_correct: bool, n_iters: int):
    print(f"\n{'='*65}")
    print(f"[{idx}/{total}] Q: {question[:100]}{'...' if len(question)>100 else ''}")
    print(f"  정답         : {gold}")
    print(f"  Self-Refine  : {sr_pred[-60:]} → {'✅' if sr_correct else '❌'} (iters: {n_iters})")
    print(f"{'='*65}")

def print_final_summary(dataset: str, sr: dict):
    print(f"\n{'='*65}")
    print(f"  [{dataset}] 최종 결과  |  모델: {MODEL_PATH.split('/')[-1]}")
    print(f"{'='*65}")
    print(f"  Self-Refine : {sr['accuracy_pct']}%  ({sr['correct']}/{sr['total']})")
    print(f"{'='*65}\n")


# ── AIME ─────────────────────────────────────────────────

def run_aime(n_samples: int = 5):
    dataset = load_aime(n_samples=n_samples)
    total   = len(dataset)
    sr_rows = []

    print(f"\n{'#'*65}")
    print(f"  AIME — {total}문제 / 모델: {MODEL_PATH.split('/')[-1]}")
    print(f"  Method: Self-Refine (max_iters={SELF_REFINE_MAX_ITER})")
    print(f"{'#'*65}")

    for idx, item in enumerate(tqdm(dataset, desc="AIME"), 1):
        q    = item["problem"]
        gold = item["answer"]

        # Self-Refine
        sr_pred, sr_logs = self_refine(q, is_mcq=False)
        sr_correct       = aime_correct(sr_pred, gold)
        n_iters          = len(sr_logs)

        print_progress(idx, total, q, gold, sr_pred, sr_correct, n_iters)

        row_base = {"idx": idx, "question": q, "gold": gold}
        sr_rows.append({**row_base,
                        "predicted": sr_pred, "correct": sr_correct,
                        "n_iters": n_iters,
                        "sr_logs": json.dumps(sr_logs, ensure_ascii=False)})

    return sr_rows


# ── MMMLU ────────────────────────────────────────────────

def run_mmmlu(n_samples: int = 50, lang: str = "ko"):
    dataset = load_mmmlu(lang=lang, n_samples=n_samples)
    total   = len(dataset)
    sr_rows = []

    print(f"\n{'#'*65}")
    print(f"  MMMLU ({lang}) — {total}문제 / 모델: {MODEL_PATH.split('/')[-1]}")
    print(f"  Method: Self-Refine (max_iters={SELF_REFINE_MAX_ITER})")
    print(f"{'#'*65}")

    for idx, item in enumerate(tqdm(dataset, desc="MMMLU"), 1):
        q_text = format_question(item)
        gold   = item["answer"]
        subj   = item["subject"]

        # Self-Refine
        sr_pred, sr_logs = self_refine(q_text, is_mcq=True)
        sr_correct       = mmmlu_correct(sr_pred, gold)
        n_iters          = len(sr_logs)

        print_progress(idx, total, item["question"], gold, sr_pred, sr_correct, n_iters)

        row_base = {"idx": idx, "subject": subj, "question": item["question"], "gold": gold}
        sr_rows.append({**row_base,
                        "predicted": sr_pred, "correct": sr_correct,
                        "n_iters": n_iters,
                        "sr_logs": json.dumps(sr_logs, ensure_ascii=False)})

    return sr_rows


# ── 저장 ─────────────────────────────────────────────────

def save_experiment(dataset: str, sr_rows: list):
    os.makedirs("results", exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = MODEL_PATH.split("/")[-1]
    base  = f"results/{dataset}_SelfRefine_{model}_{ts}"

    sr_score = score(sr_rows)

    # CSV: 문제별 상세
    save_csv(sr_rows, f"{base}_detail.csv")

    # JSON: 전체 요약 + per_question
    summary = {
        "meta": {
            "dataset":          dataset,
            "model":            model,
            "model_path":       MODEL_PATH,
            "timestamp":        ts,
            "selfrefine_iters": SELF_REFINE_MAX_ITER,
            "n_samples":        sr_score["total"],
        },
        "summary": {
            "self_refine":  sr_score
        },
        "per_question": [
            {
                "idx":              sr["idx"],
                "question":         sr["question"],
                "gold":             sr["gold"],
                "sr_pred":          sr["predicted"],
                "sr_correct":       sr["correct"],
                "sr_n_iters":       sr["n_iters"],
            }
            for sr in sr_rows
        ],
    }
    save_json(summary, f"{base}_summary.json")

    print_final_summary(dataset, sr_score)
    print(f"  저장 파일:")
    print(f"    {base}_detail.csv")
    print(f"    {base}_summary.json")

    return sr_score


# ── 메인 ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["aime", "mmmlu", "all"], default="all")
    parser.add_argument("--aime_n",  type=int, default=5)
    parser.add_argument("--mmmlu_n", type=int, default=50)
    parser.add_argument("--lang",    type=str, default="ko")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  실험 시작")
    print(f"  모델      : {MODEL_PATH}")
    print(f"  Method    : Self-Refine Only")
    print(f"  SR iters  : {SELF_REFINE_MAX_ITER}")
    print(f"{'='*65}\n")

    if args.dataset in ("aime", "all"):
        sr = run_aime(n_samples=args.aime_n)
        save_experiment("AIME", sr)

    if args.dataset in ("mmmlu", "all"):
        sr = run_mmmlu(n_samples=args.mmmlu_n, lang=args.lang)
        save_experiment("MMMLU", sr)