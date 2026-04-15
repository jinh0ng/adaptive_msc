# benchmark/evaluate_e4b.py
import json
import csv
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


# ── 유틸 ─────────────────────────────────────────────────

def score(results: list[dict]) -> dict:
    n       = len(results)
    correct = sum(r["correct"] for r in results)
    return {
        "accuracy_pct": round(correct / n * 100, 2),
        "correct":      correct,
        "total":        n,
    }

def append_csv_row(row: dict, path: str):
    """파일 없으면 헤더 포함 생성, 있으면 행만 추가"""
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def save_json(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def flush_summary_json(dataset: str, model: str, ts: str,
                       baseline_rows: list, mas_rows: list, path: str,
                       max_depth: int, max_width: int):
    """지금까지 쌓인 행으로 summary.json 전체 덮어쓰기"""
    if not baseline_rows:
        return
    b_score = score(baseline_rows)
    m_score = score(mas_rows)
    diff    = round(m_score["accuracy_pct"] - b_score["accuracy_pct"], 2)
    summary = {
        "meta": {
            "dataset":    dataset,
            "model":      model,
            "model_path": MODEL_PATH,
            "timestamp":  ts,
            "max_depth":  max_depth,
            "max_width":  max_width,
            "n_completed": b_score["total"],
        },
        "summary": {
            "baseline_accuracy": b_score["accuracy_pct"],
            "mas_accuracy":      m_score["accuracy_pct"],
            "diff_pct":          diff,
            "winner":            "MAS" if diff > 0 else ("BASELINE" if diff < 0 else "TIE"),
            "baseline_correct":  b_score["correct"],
            "mas_correct":       m_score["correct"],
            "total_completed":   b_score["total"],
        },
        "per_question": [
            {
                "idx":              b["idx"],
                "question":         b["question"],
                "gold":             b["gold"],
                "baseline_pred":    b["predicted"],
                "baseline_correct": b["correct"],
                "mas_pred":         m["predicted"],
                "mas_correct":      m["correct"],
            }
            for b, m in zip(baseline_rows, mas_rows)
        ],
    }
    save_json(summary, path)

def print_progress(idx: int, total: int, question: str,
                   gold, baseline_pred, mas_pred,
                   baseline_correct: bool, mas_correct: bool):
    """매 문제마다 터미널에 결과 출력"""
    print(f"\n{'='*65}")
    print(f"[{idx}/{total}] Q: {question[:100]}{'...' if len(question)>100 else ''}")
    print(f"  정답       : {gold}")
    print(f"  Baseline   : {str(baseline_pred)[:80]} → {'✅' if baseline_correct else '❌'}")
    print(f"  MAS        : {str(mas_pred)[:80]} → {'✅' if mas_correct else '❌'}")
    print(f"{'='*65}")

def print_final_summary(dataset: str, b_score: dict, m_score: dict):
    diff  = round(m_score["accuracy_pct"] - b_score["accuracy_pct"], 2)
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
    winner = "MAS" if diff > 0 else ("BASELINE" if diff < 0 else "TIE")
    print(f"\n{'='*65}")
    print(f"  [{dataset}] 최종 결과 요약")
    print(f"{'='*65}")
    print(f"  모델     : {MODEL_PATH.split('/')[-1]}")
    print(f"  Baseline : {b_score['accuracy_pct']}%  ({b_score['correct']}/{b_score['total']})")
    print(f"  MAS      : {m_score['accuracy_pct']}%  ({m_score['correct']}/{m_score['total']})")
    print(f"  차이     : {arrow} {abs(diff)}%p  →  {winner} 승")
    print(f"{'='*65}\n")


# ── AIME ─────────────────────────────────────────────────

def run_aime(n_samples: int = 5, base_path: str = None,
             max_depth: int = MAX_DEPTH, max_width: int = MAX_WIDTH):
    dataset = load_aime(n_samples=n_samples)
    total   = len(dataset)

    baseline_rows, mas_rows = [], []

    print(f"\n{'#'*65}")
    print(f"  AIME — {total}문제 / 모델: {MODEL_PATH.split('/')[-1]}")
    print(f"  설정: max_depth={max_depth}, max_width={max_width}")
    print(f"{'#'*65}")
    if base_path:
        print(f"  📂 실시간 저장 경로: {base_path}_*.csv / *.json\n")

    for idx, item in enumerate(tqdm(dataset, desc="AIME", unit="문제"), 1):
        q    = item["problem"]
        gold = item["answer"]

        # Baseline
        b_pred    = call_llm(build_solve_prompt(q))
        b_correct = aime_correct(b_pred, gold)

        # MAS
        agent     = RootAgent(depth=0, max_depth=max_depth, max_width=max_width)
        m_pred    = agent.run(q)
        m_correct = aime_correct(m_pred, gold)

        print_progress(idx, total, q, gold,
                       b_pred[-60:], m_pred[-60:],
                       b_correct, m_correct)

        b_row = {"idx": idx, "question": q, "gold": gold,
                 "predicted": b_pred, "correct": b_correct}
        m_row = {"idx": idx, "question": q, "gold": gold,
                 "predicted": m_pred, "correct": m_correct}

        baseline_rows.append(b_row)
        mas_rows.append(m_row)

        # ── 실시간 저장 ──────────────────────────────────
        if base_path:
            append_csv_row(b_row, f"{base_path}_baseline_detail.csv")
            append_csv_row(m_row, f"{base_path}_mas_detail.csv")
            flush_summary_json("AIME", MODEL_PATH.split("/")[-1],
                               os.path.basename(base_path).split("_")[-1],
                               baseline_rows, mas_rows,
                               f"{base_path}_summary.json",
                               max_depth, max_width)

    return baseline_rows, mas_rows


# ── MMMLU ────────────────────────────────────────────────

def run_mmmlu(n_samples: int = 50, lang: str = "ko", base_path: str = None,
              max_depth: int = MAX_DEPTH, max_width: int = MAX_WIDTH):
    dataset = load_mmmlu(lang=lang, n_samples=n_samples)
    total   = len(dataset)

    baseline_rows, mas_rows = [], []

    print(f"\n{'#'*65}")
    print(f"  MMMLU ({lang}) — {total}문제 / 모델: {MODEL_PATH.split('/')[-1]}")
    print(f"  설정: max_depth={max_depth}, max_width={max_width}")
    print(f"{'#'*65}")
    if base_path:
        print(f"  📂 실시간 저장 경로: {base_path}_*.csv / *.json\n")

    for idx, item in enumerate(tqdm(dataset, desc="MMMLU", unit="문제"), 1):
        q_text = format_question(item)
        gold   = item["answer"]
        subj   = item["subject"]

        # Baseline
        b_pred    = call_llm(build_solve_mcq_prompt(q_text))
        b_correct = mmmlu_correct(b_pred, gold)

        # MAS
        agent     = MCQAgent(depth=0, max_depth=max_depth, max_width=max_width)
        m_pred    = agent.run(q_text)
        m_correct = mmmlu_correct(m_pred, gold)

        print_progress(idx, total, item["question"], gold,
                       b_pred[-60:], m_pred[-60:],
                       b_correct, m_correct)

        b_row = {"idx": idx, "subject": subj, "question": item["question"],
                 "gold": gold, "predicted": b_pred, "correct": b_correct}
        m_row = {"idx": idx, "subject": subj, "question": item["question"],
                 "gold": gold, "predicted": m_pred, "correct": m_correct}

        baseline_rows.append(b_row)
        mas_rows.append(m_row)

        # ── 실시간 저장 ──────────────────────────────────
        if base_path:
            append_csv_row(b_row, f"{base_path}_baseline_detail.csv")
            append_csv_row(m_row, f"{base_path}_mas_detail.csv")
            flush_summary_json("MMMLU", MODEL_PATH.split("/")[-1],
                               os.path.basename(base_path).split("_")[-1],
                               baseline_rows, mas_rows,
                               f"{base_path}_summary.json",
                               max_depth, max_width)

    return baseline_rows, mas_rows


# ── 저장 (최종 완료 시) ───────────────────────────────────

def save_experiment(dataset: str, baseline_rows: list, mas_rows: list,
                    base_path: str = None, max_depth: int = MAX_DEPTH, max_width: int = MAX_WIDTH):
    """실시간 저장이 켜져 있으면 최종 summary만 재출력, 아니면 한꺼번에 저장"""
    os.makedirs("results", exist_ok=True)

    b_score = score(baseline_rows)
    m_score = score(mas_rows)

    if base_path is None:
        # 실시간 저장 없었던 경우 → 한 번에 저장
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        model = MODEL_PATH.split("/")[-1]
        base_path = f"results/{dataset}_{model}_{ts}"

        diff = round(m_score["accuracy_pct"] - b_score["accuracy_pct"], 2)
        summary = {
            "meta": {
                "dataset": dataset, "model": model, "model_path": MODEL_PATH,
                "timestamp": ts, "max_depth": max_depth, "max_width": max_width,
                "n_samples": b_score["total"],
            },
            "summary": {
                "baseline_accuracy": b_score["accuracy_pct"],
                "mas_accuracy":      m_score["accuracy_pct"],
                "diff_pct":          diff,
                "winner":            "MAS" if diff > 0 else ("BASELINE" if diff < 0 else "TIE"),
                "baseline_correct":  b_score["correct"],
                "mas_correct":       m_score["correct"],
                "total":             b_score["total"],
            },
            "per_question": [
                {"idx": b["idx"], "question": b["question"], "gold": b["gold"],
                 "baseline_pred": b["predicted"], "baseline_correct": b["correct"],
                 "mas_pred": m["predicted"], "mas_correct": m["correct"]}
                for b, m in zip(baseline_rows, mas_rows)
            ],
        }
        from benchmark.evaluate_e4b import save_json  # 자기 자신
        for rows, suffix in [(baseline_rows, "baseline_detail"),
                             (mas_rows, "mas_detail")]:
            p = f"{base_path}_{suffix}.csv"
            with open(p, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        save_json(summary, f"{base_path}_summary.json")

    print_final_summary(dataset, b_score, m_score)
    print(f"  저장된 파일:")
    print(f"    {base_path}_baseline_detail.csv")
    print(f"    {base_path}_mas_detail.csv")
    print(f"    {base_path}_summary.json")

    return b_score, m_score


# ── 메인 ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["aime", "mmmlu", "all"], default="all")
    parser.add_argument("--aime_n",  type=int, default=5)
    parser.add_argument("--mmmlu_n", type=int, default=50)
    parser.add_argument("--lang",    type=str, default="ko")
    parser.add_argument("--max_depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--max_width", type=int, default=MAX_WIDTH)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = MODEL_PATH.split("/")[-1]

    print(f"\n실험 시작")
    print(f"모델: {MODEL_PATH}")
    print(f"MAX_DEPTH={args.max_depth}, MAX_WIDTH={args.max_width}\n")

    if args.dataset in ("aime", "all"):
        base = f"results/AIME_{model}_{ts}"
        b_rows, m_rows = run_aime(n_samples=args.aime_n, base_path=base,
                                 max_depth=args.max_depth, max_width=args.max_width)
        save_experiment("AIME", b_rows, m_rows, base_path=base,
                        max_depth=args.max_depth, max_width=args.max_width)

    if args.dataset in ("mmmlu", "all"):
        base = f"results/MMMLU_{model}_{ts}"
        b_rows, m_rows = run_mmmlu(n_samples=args.mmmlu_n, lang=args.lang, base_path=base,
                                  max_depth=args.max_depth, max_width=args.max_width)
        save_experiment("MMMLU", b_rows, m_rows, base_path=base,
                        max_depth=args.max_depth, max_width=args.max_width)