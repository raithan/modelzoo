import timm

# 列出所有模型
all_models = timm.list_models()
print(all_models)

# 只列出 BEiT 系列模型
beit_models = timm.list_models('*beit*')
print(beit_models)
