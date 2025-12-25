from funasr import AutoModel

# 设置本地模型路径
model_dir = "/data/bigc-data/zh/FunASR-main/model_zoo"

# 加载本地模型
model = AutoModel(
    model=f"{model_dir}/paraformer-zh.pt",     # ASR 主模型
    vad_model=f"{model_dir}/fsmn-vad.pt",      # VAD 模型
    punc_model=f"{model_dir}/ct-punc.pt",      # 标点恢复模型
)

# 执行推理
res = model.generate(input="asr_example_zh.wav")

# 输出结果
print(res)