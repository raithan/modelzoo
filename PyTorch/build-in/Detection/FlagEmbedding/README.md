# bge-large-zh 微调与使用指南

本指南介绍如何在本地环境下使用 **bge-large-zh**（1.5 版本示例）进行：

1. 模型准备
2. 数据集构造（检索三元组 / Hard Negative）
3. 100 步快速验证微调
4. 评估与日志记录
5. 进阶：Hard Negative 迭代、参数调优、典型问题排查

> 目标读者：需要快速落地中文语义检索微调（query → 文档向量）与部署的工程人员。

---

## 1. 模型准备

### 1.1 获取模型文件

从官方仓库下载或在已有缓存中整理到一个目录，例如：

```
/path/to/bge-large-zh-v1.5/
  ├─ config.json
  ├─ tokenizer.json
  ├─ tokenizer_config.json
  ├─ vocab.txt
  ├─ model.safetensors   (推荐，如只有 pytorch_model.bin 可先转换)
```

如只有 `pytorch_model.bin`，可用：

```python
import torch, safetensors.torch
sd = torch.load("pytorch_model.bin", map_location="cpu")
safetensors.torch.save_file(sd, "model.safetensors", metadata={"format":"pt"})
```

然后可将 `pytorch_model.bin` 备份为 `pytorch_model.bin.bak`，默认优先加载 `model.safetensors`。

### 1.2 目录结构建议

```
project_root/
  ├─ models/bge-large-zh-v1.5/
  ├─ data/
  │   ├─ zh_triplets_posneg.jsonl
  │   └─ zh_triplets_posneg_hard.jsonl (可选)
  ├─ scripts/
  │   ├─ train_bge_large_zh_100steps.py
  │   └─ build_hard_negatives.py
  └─ outputs/ (训练结果与日志)
```

---

## 2. 数据集构造

### 2.1 基础三元组格式

`jsonl` 每行一个样本：

```json
{"query":"为这个句子生成表示以用于检索相关文章：人工智能赋能产业升级","pos":["人工智能正在通过…"],"neg":["另一篇不相关文本…"]}
```

字段说明：

* **query**：检索请求（可添加指令前缀以统一风格）
* **pos**：正例文本列表（通常 1 条或多条）
* **neg**：负例文本列表（≥1 条；建议包含随机 + hard）

### 2.2 构造基础随机负例流程（示意）

```python
import json, random
random.seed(42)
all_texts = [...]  # 文档集合 / 正文数组
prefix = "为这个句子生成表示以用于检索相关文章："
triplets = []
for doc in all_texts:
    if len(doc) < 15: continue
    query = prefix + doc[:30]
    neg = random.choice(all_texts)
    while neg == doc:
        neg = random.choice(all_texts)
    triplets.append({"query":query, "pos":[doc], "neg":[neg]})
with open("data/zh_triplets_posneg.jsonl","w",encoding="utf-8") as f:
    for t in triplets: f.write(json.dumps(t, ensure_ascii=False) + "\n")
```

> 随机负例简单但容易使损失过早饱和，需后续加入 hard negatives。

### 2.3 生成 Hard Negatives（静态）

1. 用初始模型对所有 `pos` 文本编码（归一化向量）。
2. 计算相似度矩阵（或用 Faiss ANN）。
3. 对每条样本挑选若干最相似且非自身的文本作为 hard negatives，写回新文件：`zh_triplets_posneg_hard.jsonl`。

示意脚本核心：

```python
# 假设已有 base_triplets 列表，其中每条含 pos[0]
embs = encode([t['pos'][0] for t in base_triplets])   # (N, d)
sim = embs @ embs.T
np.fill_diagonal(sim, -1)
K = 10
for i,t in enumerate(base_triplets):
    idx = sim[i].argsort()[-K:][::-1]  # 相似度降序
    hard_list = []
    for j in idx:
        cand = base_triplets[j]['pos'][0]
        if cand != t['pos'][0]:
            hard_list.append(cand)
        if len(hard_list) >= 3: break
    t['neg'] = hard_list
```

> 若样本数量 >50k，建议使用 Faiss（IVF / HNSW）边编码边检索。

### 2.4 动态迭代挖掘（可选）

* 初始：随机负例训练少量步数
* 第 1 轮：用当前模型挖 hard negatives → 重新训练
* 第 2 轮：进一步在 hard 基础上混合 harder（高相似度阈值）负例

---

## 3. 100 步快速验证微调

### 3.1 自定义最小训练脚本要点

* 取 `query`, `pos`, `neg` 组装 batch
* 编码并池化（`mean pooling` + L2 normalize）
* 构建 InfoNCE / 对比损失：正例得分 vs (批内其它 + 指定负例)
* 控制步数 `--steps 100` 输出固定格式日志：`[epoch 1/1 | step X/100] loss ...`

### 3.2 示例命令

```
python scripts/train_bge_large_zh_100steps.py \
  --model_dir models/bge-large-zh-v1.5 \
  --data_path data/zh_triplets_posneg_hard.jsonl \
  --steps 100 \
  --batch_size 4 \
  --neg_each 3 \
  --lr 5e-5 \
  --temperature 0.15 \
  --fp16
```

> 若 loss 贴近 0，说明负例过易或温度过低：提高温度 (0.15→0.2)，增加 hard 数量，或提升相似度阈值。

### 3.3 常见损失函数变体

| 名称                 | 说明                    | 适用场景       |
| ------------------ | --------------------- | ---------- |
| InfoNCE (in-batch) | 正例 vs 批内所有其它作为负例      | 批内多样性高时有效  |
| 多正例对比              | 同一 query 的多正例平均       | 多视角 / 同义表达 |
| Margin Ranking     | max(0, m + neg - pos) | 控制间隔，防过饱和  |
| 双向 InfoNCE         | query→doc + doc→query | 双向检索一致性    |

---

## 4. 评估与日志

### 4.1 简易在线评估（开发阶段）

* 取一小批查询，建立候选库（正例 + 若干干扰）
* 计算向量，做余弦排序，统计 Recall\@K / MRR\@K

示意：

```python
import numpy as np
# embs_q: (Q, d)  embs_corpus: (C, d)
S = embs_q @ embs_corpus.T
# 对每个 q 的正例位置计算指标
```

### 4.2 日志提取与可视化

可统一记录：

```
[epoch 1/1 | step 10/100] loss 1.2345 pos_sim 0.62 neg_sim 0.41
```

然后使用比较脚本（如 `compare_loss.py`）生成：

* `loss_compare.png`
* `loss_compare.csv`
* 指标 JSON (MeanRelativeError, MeanAbsoluteError)

> 当需要对比两种加速后端或不同运行环境的训练曲线时，保持同一解析格式即可。

### 4.3 指标解释

| 指标                | 定义                 | 注意             |     |   |
| ----------------- | ------------------ | -------------- | --- | - |
| MeanRelativeError | mean((A-B)/B) 或自定义 | 需确认分母选项一致      |     |   |
| MeanAbsoluteError | mean(A-B)          | 若需绝对值请使用 mean( | A-B | ) |
| Recall\@K         | 正例进入前 K 的比例        | K 常用 1/5/10    |     |   |
| MRR\@K            | 平均 1/排名            | 排名> K 视为 0     |     |   |

---

## 5. 进阶技巧

### 5.1 Query 改写策略（减少过拟合前缀）

* 子串截取 → 随机删除部分 token → 关键词抽取 → 问句化（"这段文本的主题是什么？" + 摘要）
* 同义替换：可用同义词词典或轻量生成模型批量改写

### 5.2 Hard Negative 多样化

| 类型         | 获取方式        | 价值          |
| ---------- | ----------- | ----------- |
| 语义近邻       | 向量近邻        | 主力：提升判别能力   |
| BM25 高分非正例 | 传统检索        | 覆盖词面相似但语义偏差 |
| 跨域近邻       | 不同数据子域相似文本  | 稳定泛化        |
| 对抗改写       | 轻度打乱 / 替换实体 | 提升鲁棒性       |

### 5.3 学习率与温度调优

| 症状                               | 处理                              |
| -------------------------------- | ------------------------------- |
| Loss 迅速趋 0 且 pos\_sim ≫ neg\_sim | 增加 hard 数量 / 提高温度 / margin loss |
| Loss 长期不降                        | 降温度或调低学习率 / 检查负例是否过难            |
| 训练震荡                             | 减小 batch 或学习率，或使用梯度裁剪           |

### 5.4 训练规模放大建议

| 目标     | 调整                                         |
| ------ | ------------------------------------------ |
| 更高检索精度 | 组合多来源语料 + 迭代 hard mining                   |
| 更快收敛   | 利用多 GPU 的 in-batch negatives，提高 batch size |
| 更强泛化   | 混合多领域（科技、金融、百科、对话）文本                       |

### 5.5 混合多正例

若一个 query 对应多个等价段落：将 `pos` 列表全部纳入同一 batch 样本，或展开为多行再做去重。

---

## 6. 常见问题排查

| 问题              | 现象                              | 解决                          |
| --------------- | ------------------------------- | --------------------------- |
| 路径错误            | "Incorrect path\_or\_model\_id" | 指向含 config.json 的目录         |
| 权重格式报风险         | 加载失败或安全限制                       | 转换为 model.safetensors       |
| Loss 恒为 0       | 负例过易 / 温度过低                     | Hard Neg + 提升温度 (0.15\~0.2) |
| Loss 不下降        | 负例过难或数据噪声                       | 混合随机与 hard；调低难度阈值           |
| KeyError: 'pos' | 数据行字段不规范                        | 校验生成脚本，保证 pos/neg 列表        |
| 内存 / 显存不足       | 大规模一次矩阵相似度                      | 分块或使用 Faiss ANN             |
| 检索效果差           | Recall\@K 低                     | 提升数据多样性 + 迭代挖掘              |

---

## 7. 示例：完整最小训练脚本骨架（提炼版）

```python
import json, random, torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

class TripletDS(Dataset):
    def __init__(self, path):
        self.samples=[]
        with open(path,'r',encoding='utf-8') as f:
            for l in f:
                if not l.strip(): continue
                o=json.loads(l)
                if 'query' in o and 'pos' in o and 'neg' in o:
                    self.samples.append(o)
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        o=self.samples[i]
        return o['query'], o['pos'][0], o['neg']

def mean_pool(last_hidden, mask):
    mask=mask.unsqueeze(-1).type_as(last_hidden)
    return (last_hidden*mask).sum(1)/mask.sum(1).clamp(min=1e-6)

def collate(batch, tok, q_max=64, p_max=512):
    qs,ps,negs = zip(*batch)
    flat_negs=[n for sub in negs for n in sub]
    q_tok=tok(list(qs),padding=True,truncation=True,max_length=q_max,return_tensors='pt')
    p_tok=tok(list(ps),padding=True,truncation=True,max_length=p_max,return_tensors='pt')
    n_tok=tok(flat_negs,padding=True,truncation=True,max_length=p_max,return_tensors='pt')
    offs=[];c=0
    for sub in negs:
        offs.append((c,c+len(sub)));c+=len(sub)
    return q_tok,p_tok,n_tok,offs

def contrastive(q,p,n,offs,temp=0.15):
    B=q.size(0)
    # in-batch + explicit negatives
    pos_logits = (q@p.t())/temp
    rows=[]
    for i in range(B):
        st,ed=offs[i]
        extra = (q[i:i+1]@n[st:ed].t())/temp
        rows.append(torch.cat([pos_logits[i], extra.squeeze(0)],0).unsqueeze(0))
    logits=torch.cat(rows,0)
    labels=torch.arange(B,device=q.device)
    return nn.CrossEntropyLoss()(logits,labels)

# main 省略 (加载 tokenizer/model, DataLoader, loop, 打印日志)
```

---

## 8. 后续扩展方向

* 引入对比蒸馏（teacher embedding 监督）
* 多阶段 curriculum：随机负例 → 半硬 → 硬
* 混合多任务：句子相似度（STS）、问答对、百科段落检索
* 语义去偏：实体随机替换 + 不改变标签增强鲁棒性

---