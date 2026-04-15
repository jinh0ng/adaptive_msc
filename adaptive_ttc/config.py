# config.py
# 모델 갈아끼울 때 여기 한 줄만 수정

MODEL_PATH = "/data/yejinhong/gemma-4-E2B-it"
# MODEL_PATH = "/data/yejinhong/gemma-4-E4B-it"   # 나중에 교체용
#MODEL_PATH = "/data/yejinhong/gemma-4-E4B-it"

# 트리 설정
MAX_DEPTH = 2
MAX_WIDTH = 3

# 생성 설정
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0

# 실험 샘플 수
N_SAMPLES_QUICK = 3
N_SAMPLES_FULL  = 100
