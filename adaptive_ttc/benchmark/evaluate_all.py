# benchmark/evaluate_all.py
import json
import os
from datetime import datetime
from tqdm import tqdm

from llm import call_llm
from agents.root_agent import RootAgent
from agents.mcq_agent import MCQAgent
from benchmark.gsm8k import load_gsm8k, is_correct as gsm_correct
from benchmark.aime import load_aime, is_correct as aime_correct
from benchmark.mmmlu import load_mmmlu, format_question, is_correct as mmmlu_correct
from prompts.solve import build_solve_prompt
from prompts.solve_mcq import build_solve_mcq_prompt
from config import MODEL_PATH, MAX_DEPTH, MAX_WIDTH


# ── Baseline 함수들 ──────────────────────────────────────

def baseline_gsm8k(dataset):
    results = []
    for item in tqdm(dataset, desc="[Baseline] GSM8K"):
        pred = call_llm(build_solve_prompt(item["question"]))
        correct = gsm_correct(pred, item["answer"])
        results.append({
            "question": item["question"],
            "gold": item["answer"],
            "predicted": pred,
            "correct": correct,
        })
    return results

def baseline_aime(dataset):
    results = []
    for item in tqdm(dataset, desc="[Baseline] AIME"):
        pred = call_llm(build_solve_prompt(item["problem"]))
        correct = aime_correct(pred, item["answer"])
        results.append({
            "question": item["problem"],
            "gold": item["answer"],
            "predicted": pred,
            "correct": correct,
        })
    return results

def baseline_mmmlu(dataset):
    results = []
    for item in tqdm(dataset, desc="[Baseline] MMMLU"):
        q = format_question(item)
        pred = call_llm(build_solve_mcq_prompt(q))
        correct = mmmlu_correct(pred, item["answer"])
        results.append({
            "question": item["question"],
            "gold": item["answer"],
            "subject": item["subject"],
            "predicted": pred,
            "correct": correct,
        })
    return results


# ── MAS 함수들 ───────────────────────────────────────────

def mas_gsm8k(dataset):
    results = []
    for item in tqdm(dataset, desc="[MAS] GSM8K"):
        agent = RootAgent(depth=0)
        pred = agent.run(item["question"])
        correct = gsm_correct(pred, item["answer"])
        results.append({
            "question": item["question"],
            "gold": item["answer"],
            "predicted": pred,
            "correct": correct,
        })
    return results

def mas_aime(dataset):
    results = []
    for item in tqdm(dataset, desc="[MAS] AIME"):
        agent = RootAgent(depth=0)
        pred = agent.run(item["problem"])
        correct = aime_correct(pred, item["answer"])
        results.append({
            "question": item["problem"],
            "gold": item["answer"],
            "predicted": pred,
            "correct": correct,
        })
    return results

def mas_mmmlu(dataset):
    results = []
    for item in tqdm(dataset, desc="[MAS] MMMLU"):
        q = format_question(item)
        agent = MCQAgent(depth=0)
        pred = agent.run(q)
        correct = mmmlu_correct(pred, item["answer"])
        results.append({
            "question": item["question"],
            "gold": item["answer"],
            "subject": item["subject"],
            "predicted": pred,
            "correct": correct,
        })
    return results


# ── 유틸 ─────────────────────────────────────────────────

def score(results):
    n = len(results)
    correct = sum(r["correct"] for r in results)
    return {"accuracy": round(correct / n * 100, 2), "correct": correct, "total": n}

def save_and_print(name, baseline_r, mas_r, mode):
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    b = score(baseline_r)
    m = score(mas_r)
    diff = round(m["accuracy"] - b["accuracy"], 2)

    output = {
        "meta": {
            "dataset": name,
            "mode": mode,
            "model": MODEL_PATH,
            "max_depth": MAX_DEPTH,
            "max_width": MAX_WIDTH,
            "timestamp": ts,
        },
        "summary": {
            "baseline_accuracy": b["accuracy"],
            "mas_accuracy": m["accuracy"],
            "diff": diff,
            "winner": "MAS" if diff > 0 else ("BASELINE" if diff < 0 else "TIE"),
        },
        "baseline": {"score": b, "results": baseline_r},
        "mas":      {"score": m, "results": mas_r},
    }

    path = f"results/{name}_{mode}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
    print(f"\n{'='*50}")
    print(f"  [{name}] 결과 요약")
    print(f"{'='*50}")
    print(f"  Baseline : {b['accuracy']}%  ({b['correct']}/{b['total']})")
    print(f"  MAS      : {m['accuracy']}%  ({m['correct']}/{m['total']})")
    print(f"  차이     : {arrow} {abs(diff)}%p  → {output['summary']['winner']} 승")
    print(f"  저장     : {path}")
    print(f"{'='*50}")


# ── 메인 ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                        help="quick=3개 / full=100개 (AIME는 전체 30개)")
    parser.add_argument("--dataset", choices=["gsm8k", "aime", "mmmlu", "all"],
                        default="all")
    args = parser.parse_args()

    n_quick = 3
    n_full  = 100

    print(f"\n모델  : {MODEL_PATH}")
    print(f"모드  : {args.mode}")
    print(f"대상  : {args.dataset}\n")

    # GSM8K
    # if args.dataset in ("gsm8k", "all"):
    #     n = n_quick if args.mode == "quick" else n_full
    #     data = load_gsm8k(split="test", n_samples=n)
    #     b = baseline_gsm8k(data)
    #     m = mas_gsm8k(data)
    #     save_and_print("GSM8K", b, m, args.mode)
    # MMMLU
    if args.dataset in ("mmmlu", "all"):
        n = n_quick if args.mode == "quick" else n_full
        data = load_mmmlu(lang="ko", n_samples=n)
        b = baseline_mmmlu(data)
        m = mas_mmmlu(data)
        save_and_print("MMMLU", b, m, args.mode)
        
    # AIME
    if args.dataset in ("aime", "all"):
        # AIME는 총 30개 → quick=3, full=전체 30개
        n = n_quick if args.mode == "quick" else None
        data = load_aime(n_samples=n)
        b = baseline_aime(data)
        m = mas_aime(data)
        save_and_print("AIME", b, m, args.mode)

