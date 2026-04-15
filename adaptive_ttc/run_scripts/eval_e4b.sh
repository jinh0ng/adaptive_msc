#!/bin/bash
#SBATCH --job-name=ttc_e4b
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --nodelist=n01
#SBATCH --output=/data/yejinhong/adaptive_ttc/slurm_output/%x_%j.out
#SBATCH --error=/data/yejinhong/adaptive_ttc/slurm_output/%x_%j.err

# ── Environment Setup ─────────────────────────────────────
source /data/${USER}/.bashrc
source /data/yejinhong/miniconda3/etc/profile.d/conda.sh
conda activate ttc

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

export HF_HOME=/data/yejinhong/data/hub
export HF_TOKEN="YOUR TOKEN "

# ── Project Root ──────────────────────────────────────────
PROJECT_ROOT="/data/yejinhong/adaptive_ttc"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
cd "$PROJECT_ROOT"

# ── Output 디렉토리 ──────────────────────────────────────
if [[ -n "${SLURM_JOB_NAME:-}" && -n "${SLURM_JOB_ID:-}" ]]; then
    SLURM_OUT_DIR="${PROJECT_ROOT}/slurm_output/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SLURM_OUT_DIR="${PROJECT_ROOT}/slurm_output/job_${SLURM_JOB_ID}"
else
    SLURM_OUT_DIR="${PROJECT_ROOT}/slurm_output/local_eval"
fi
mkdir -p "$SLURM_OUT_DIR"

# ── 데이터셋 경로 확인 ────────────────────────────────────
AIME_PATH="${PROJECT_ROOT}/dataset/aime_2026/data/train-00000-of-00001.parquet"
MMMLU_PATH="${PROJECT_ROOT}/dataset/MMMLU/test/mmlu_KO-KR.csv"

echo "========================================================="
echo "  Job: ${SLURM_JOB_NAME:-local}  ID: ${SLURM_JOB_ID:-none}"
echo "  Node: $(hostname)"
echo "  Start: $(date)"
echo "  Project: $PROJECT_ROOT"
echo "  Output dir: $SLURM_OUT_DIR"
echo "========================================================="

for f in "$AIME_PATH" "$MMMLU_PATH"; do
    if [[ ! -f "$f" ]]; then
        echo "❌ 필수 데이터 파일 없음: $f"
        exit 1
    fi
done
echo "✅ 데이터셋 파일 확인 완료"

# ── E4B 모델 다운로드 (없을 경우 자동 clone) ─────────────
MODEL_DIR="/data/yejinhong/gemma-4-E4B-it"
# if [[ -d "$MODEL_DIR" && -f "$MODEL_DIR/config.json" ]]; then
#     echo "✅ E4B 모델 이미 존재: $MODEL_DIR (skip clone)"
# else
#     echo "📥 E4B 모델 클론 시작..."
#     git clone https://huggingface.co/google/gemma-4-E4B-it "$MODEL_DIR"
#     echo "✅ E4B 모델 클론 완료: $(date)"
# fi

# ── 실험 실행 ─────────────────────────────────────────────
echo ""
echo "🚀 실험 시작: E4B 모델 / AIME 5문제 + MMMLU 50문제 (Baseline vs MAS)"
echo ""

python3 benchmark/evaluate_e4b.py \
    --dataset all \
    --aime_n 5 \
    --mmmlu_n 30

echo ""
echo "✅ 실험 완료: $(date)"
echo "   결과 저장 위치: ${PROJECT_ROOT}/results/"
