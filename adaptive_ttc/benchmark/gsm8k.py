# benchmark/gsm8k.py
import re
import pandas as pd

TRAIN_PATH = "/data/yejinhong/adaptive_ttc/dataset/gsm8k/main/train-00000-of-00001.parquet"
TEST_PATH  = "/data/yejinhong/adaptive_ttc/dataset/gsm8k/main/test-00000-of-00001.parquet"

def load_gsm8k(split: str = "test", n_samples: int = None) -> list[dict]:
    """
    split: "train" or "test"
    n_samples: None이면 전체, 숫자면 앞에서 n개만
    반환: [{"question": ..., "answer": ...}, ...]
    """
    path = TEST_PATH if split == "test" else TRAIN_PATH
    df = pd.read_parquet(path)
    if n_samples is not None:
        df = df.iloc[:n_samples]
    return df[["question", "answer"]].to_dict(orient="records")


def extract_gold_answer(answer_text: str) -> float | None:
    """
    GSM8K 정답 텍스트에서 #### 뒤 숫자 추출
    예: "#### 72" → 72.0
    """
    match = re.search(r"####\s*([\d,\-\.]+)", answer_text)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def extract_pred_answer(pred_text: str) -> float | None:
    """
    모델 출력에서 숫자 추출
    우선순위: "ANSWER: 72" → 마지막 숫자
    """
    # 1순위: "ANSWER: 숫자"
    match = re.search(r"ANSWER:\s*([\d,\-\.]+)", pred_text)
    if match:
        return float(match.group(1).replace(",", ""))
    # 2순위: 텍스트 마지막에 나오는 숫자
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", pred_text)
    if nums:
        return float(nums[-1].replace(",", ""))
    return None


def is_correct(pred_text: str, gold_text: str) -> bool:
    pred = extract_pred_answer(pred_text)
    gold = extract_gold_answer(gold_text)
    if pred is None or gold is None:
        return False
    return abs(pred - gold) < 1e-3