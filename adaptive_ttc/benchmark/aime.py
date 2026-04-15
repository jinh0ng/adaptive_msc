# benchmark/aime.py
import re
import pandas as pd

AIME_PATH = "/data/yejinhong/adaptive_ttc/dataset/aime_2026/data/train-00000-of-00001.parquet"

def load_aime(n_samples: int = None) -> list[dict]:
    df = pd.read_parquet(AIME_PATH)
    if n_samples is not None:
        df = df.iloc[:n_samples]
    return df[["problem", "answer"]].to_dict(orient="records")

def extract_pred_answer(pred_text: str) -> float | None:
    # 1순위: "ANSWER: 숫자"
    match = re.search(r"ANSWER:\s*([\d,\-\.]+)", pred_text)
    if match:
        val = match.group(1).replace(",", "").strip()
        if val:
            return float(val)

    # 2순위: 마지막 숫자 (빈 문자열 필터링 추가)
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", pred_text)
    nums = [n for n in nums if n.strip()]  # 빈 문자열 제거
    if nums:
        return float(nums[-1].replace(",", ""))

    return None

def is_correct(pred_text: str, gold_answer) -> bool:
    pred = extract_pred_answer(pred_text)
    if pred is None:
        return False
    return abs(pred - float(gold_answer)) < 1e-3