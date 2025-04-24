export CUDA_VISIBLE_DEVICES=4,5
vllm serve \
    "/data/zeju/DeepSeek-R1-Distill-Qwen-14B" \
    --served-model-name "DeepSeek-R1-Distill-Qwen-14B" \
    --port 8016 \
    --tensor-parallel-size 2 \
    --dtype auto \
    --api-key "token-abc123" \
    --gpu_memory_utilization 0.9 \
    # --enable-prefix-caching
