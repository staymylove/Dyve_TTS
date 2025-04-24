
K=8
DEBUG=false
RESUME=false
BENCHMARK="math"
BASE_MODEL_NAME="DeepSeek-R1-Distill-Qwen-14B"
PRM_NAME="deepseek-r1-14b-cot-math-reasoning-full"


python3 ./evaluate_bon_rm.py --k "$K" --debug "$DEBUG" --resume "$RESUME" --benchmark "$BENCHMARK" --base_model_name "$BASE_MODEL_NAME" --PRM_name "$PRM_NAME"
