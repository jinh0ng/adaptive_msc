# benchmark/mmmlu.py
import re
import pandas as pd
import os

MMMLU_DIR = "/data/yejinhong/adaptive_ttc/dataset/MMMLU/test"

# 사용 가능한 언어 파일 목록
AVAILABLE_LANGS = {
    "ko": "mmlu_KO-KR.csv",
    "en": None,   # 영어는 별도 파일 없음 (KO-KR 등으로 대체)
    "ja": "mmlu_JA-JP.csv",
    "zh": "mmlu_ZH-CN.csv",
    "fr": "mmlu_FR-FR.csv",
    "de": "mmlu_DE-DE.csv",
}

def load_mmmlu(lang: str = "ko", n_samples: int = None) -> list[dict]:
    filename = AVAILABLE_LANGS.get(lang)
    if filename is None:
        raise ValueError(f"지원하지 않는 언어: {lang}. 사용 가능: {list(AVAILABLE_LANGS.keys())}")
    
    path = os.path.join(MMMLU_DIR, filename)
    df = pd.read_csv(path)
    if n_samples is not None:
        df = df.iloc[:n_samples]

    records = []
    for _, row in df.iterrows():
        records.append({
            "question": row["Question"],
            "A": row["A"],
            "B": row["B"],
            "C": row["C"],
            "D": row["D"],
            "answer": row["Answer"],      # "A", "B", "C", "D" 중 하나
            "subject": row["Subject"],
        })
    return records

def format_question(item: dict) -> str:
    """객관식 문제를 모델 입력 형식으로 변환"""
    return (
        f"{item['question']}\n\n"
        f"A) {item['A']}\n"
        f"B) {item['B']}\n"
        f"C) {item['C']}\n"
        f"D) {item['D']}"
    )

def extract_pred_answer(pred_text: str) -> str | None:
    """
    모델 출력에서 A/B/C/D 추출
    우선순위: "ANSWER: A" → 텍스트 내 단독 A/B/C/D
    """
    # 1순위: "ANSWER: A"
    match = re.search(r"ANSWER:\s*([ABCD])", pred_text.upper())
    if match:
        return match.group(1)
    # 2순위: 마지막으로 등장하는 단독 A/B/C/D
    matches = re.findall(r"\b([ABCD])\b", pred_text.upper())
    if matches:
        return matches[-1]
    return None

def is_correct(pred_text: str, gold_answer: str) -> bool:
    pred = extract_pred_answer(pred_text)
    if pred is None:
        return False
    return pred.upper() == gold_answer.upper()