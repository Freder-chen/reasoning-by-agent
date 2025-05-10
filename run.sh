# Start the vLLM server first

# model_name_or_path="Qwen/Qwen2.5-32B-Instruct"
# vllm serve ${model_name_or_path} \
#     --port 8009 \
#     --served-model-name qwen_instruct \
#     --tensor-parallel-size 2 \
#     --gpu-memory-utilization 0.9 \
#     --enable-auto-tool-choice --tool-call-parser hermes

# After starting the server, run the client script
python demo.py
