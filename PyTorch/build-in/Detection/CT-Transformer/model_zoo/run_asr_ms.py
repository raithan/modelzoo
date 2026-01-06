from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 替换为你本地 ModelScope 格式的模型路径
asr_pipeline = pipeline(
    task=Tasks.asr,
    model='./model_zoo/paraformer-zh',
    model_revision='summary'  # 或 'master'
)

# 推理
result = asr_pipeline("asr_example_zh.wav")

# 输出结果
print(result["text"])