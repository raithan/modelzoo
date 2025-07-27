# bge-m3 微调与使用指南

本 README 说明如何在本地/离线环境中完成 **bge-m3** 的：

1. 模型权重准备（GGUF → safetensors）
2. 数据集构造与格式要求（query/pos/neg 三元组）
3. 100 步快速微调示例（含日志输出）
4. 训练后简单验证与可视化
5. 常见问题排查与进阶优化

---

## 1. 模型准备

### 1.1 获取 GGUF 权重

从 Hugging Face 下载 16-bit GGUF：

```bash
mkdir -p models/bge-m3
wget -O models/bge-m3/bge-m3-f16.gguf \
  https://huggingface.co/BAAI/bge-m3-gguf/resolve/main/bge-m3-f16.gguf
```

> 8bit/4bit 量化权重一般只适合推理，不推荐再训练。

### 1.2 GGUF → PyTorch safetensors

使用转换脚本把 GGUF 转成 `model.safetensors`，并导出 `config.json`、`tokenizer.json`：

```bash
python convert_gguf_to_safetensors.py \
  --gguf models/bge-m3/bge-m3-f16.gguf \
  --out-dir models/bge-m3
```

转换完成后目录应包含：

```
models/bge-m3/
  ├─ bge-m3-f16.gguf
  ├─ model.safetensors
  ├─ config.json
  ├─ tokenizer.json
  ├─ tokenizer_config.json        (若缺失可手动创建)
  ├─ special_tokens_map.json      (最小示例如下)
```

最小 `special_tokens_map.json`：

```json
{
  "unk_token": "[UNK]",
  "sep_token": "[SEP]",
  "pad_token": "[PAD]",
  "cls_token": "[CLS]",
  "mask_token": "[MASK]"
}
```

---

## 2. 数据集准备

### 2.1 三元组 JSONL 格式

每行一个 JSON，必须包含 `query`、`pos`、`neg` 字段：

```json
{"query": "为这个句子生成表示以用于检索相关文章：新能源汽车成本优势", "pos": ["正例文本"], "neg": ["负例文本1", "负例文本2"]}
```

* **query**：查询文本（可加统一前缀，例如“为这个句子生成表示以用于检索相关文章：”）
* **pos**：正例列表（通常 1 条）
* **neg**：负例列表（≥1 条）

### 2.2 随机负例 vs Hard Negative

* 随机负例：从全集随机取不相关文本，简单，但容易 loss 过早饱和。
* Hard Negative：用当前模型/其它模型检索最相似但错误的文本，训练更有效。

> 推荐先用随机负例完成冒烟测试（100 步），随后迭代加入 Hard Negative。

---

## 3. 100 步快速微调

### 3.1 脚本入口

确保已有 `train_bge_m3.py`（示例功能：mean pooling + L2、InfoNCE 损失、固定步数日志输出）。

### 3.2 启动命令示例

```bash
python -u train_bge_m3.py \
  --model_dir models/bge-m3 \
  --data_path data/zh_triplets_posneg_hard.jsonl \
  --steps 100 \
  --batch_size 4 \
  --neg_each 3 \
  --lr 5e-5 \
  --temperature 0.15 \
  --fp16 \
  --save_dir outputs/bge_m3_ft \
  > sdaa_loss.log 2>&1 &

tail -f sdaa_loss.log
```

日志输出格式示例：

```
[epoch 1/1 | step 10/100] loss 3.9857
...
[epoch 1/1 | step 100/100] loss 2.9483
```

> 若终端“无输出”，是因为已重定向到日志，可用 `tail -f` 实时查看。

---

## 4. 训练后验证与可视化

### 4.1 简单相似度检查

随机抽样三元组，编码后打印 `cos(q,pos)` 与 `cos(q,neg)`，正例相似度应显著高于负例。

```python
q_emb = encode(q)   # L2 normalize
p_emb = encode(pos)
n_emb = encode(neg)
print("cos(q,pos)=", (q_emb@p_emb.T).item())
print("cos(q,neg)=", (q_emb@n_emb.T).item())
```

### 4.2 Loss 曲线与误差指标

可使用脚本对比不同环境的 loss：

* 输入：`cuda_loss.log` / `sdaa_loss.log`
* 输出：`loss.jpg`、`MeanRelativeError`、`MeanAbsoluteError`

命令示例：

```bash
python compare_loss.py \
  --sdaa-log sdaa_loss.log \
  --cuda-log cuda_loss.log
```

---

## 5. 常见问题排查

| 问题/症状                  | 可能原因                                            | 处理办法                                   |
| ---------------------- | ----------------------------------------------- | -------------------------------------- |
| `Can't load tokenizer` | 缺失 `tokenizer.json` 或 `special_tokens_map.json` | 补齐文件；确认目录路径正确                          |
| `config.json` 解析失败     | 文件为空/损坏                                         | 重新生成或手写最小配置；保证与权重维度一致                  |
| loss≈ln(类别数) 长期不降      | 负例过易/温度过低/参数没更新                                 | 增加 hard neg、调高 `temperature`、检查梯度是否为 0 |
| loss 迅速→0              | 负例过易且温度太低                                       | 增加难负例、提高温度或使用 margin loss              |
| KeyError: 'pos'        | 数据格式不对                                          | 检查 jsonl，确保 `query/pos/neg` 字段齐全       |
| 训练无输出                  | 输出全重定向                                          | 使用 `-u` 或 `tail -f` 观察实时日志             |

---

## 6. 进阶优化建议

* **动态 Hard Mining**：训练过程中定期用当前模型检索新的 hard negatives。
* **双向对比**：query→doc + doc→query 双向 InfoNCE 提升一致性。
* **多任务混合**：加入 STS / NLI / QA 数据，增强语义泛化。
* **数据增强**：查询改写、实体替换、句子裁剪等方式增加多样性。
* **梯度裁剪 / warmup**：防止初期梯度爆炸或震荡。

---

## 7. 附录：GGUF → safetensors 通用转换脚本

```python
#!/usr/bin/env python3
import argparse, os, json, pathlib, torch, safetensors.torch, importlib
from tqdm import tqdm

def yield_tensors(reader):
    ts = reader.tensors
    if isinstance(ts, dict):
        for k,v in ts.items():
            yield k,v
    else:
        for item in ts:
            if isinstance(item, tuple) and len(item)==2:
                yield item
            else:
                name = getattr(item, 'name', None)
                nd   = getattr(item, 'tensor', None) or getattr(item, 'data', None)
                if name is None or nd is None:
                    raise TypeError(f"Unknown tensor structure: {type(item)}")
                yield name, nd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    gguf = importlib.import_module('gguf')
    reader = gguf.GGUFReader(args.gguf)

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    state = {}
    for name, nd in tqdm(list(yield_tensors(reader)), unit='tensor'):
        state[name] = torch.tensor(nd, dtype=torch.float16)

    safetensors.torch.save_file(state, os.path.join(args.out_dir, 'model.safetensors'),
                                metadata={'format':'pt'})

    cfg = {}
    if hasattr(reader, 'metadata') and isinstance(reader.metadata, dict):
        cfg = reader.metadata.get('config', {}) or {}
    with open(os.path.join(args.out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    tok_path = os.path.join(args.out_dir, 'tokenizer.json')
    if hasattr(reader, 'write_tokenizer_json'):
        reader.write_tokenizer_json(tok_path)

    print('Done.')

if __name__ == '__main__':
    main()
```

---

## 8. 总结

* **数据质量和负例难度** 决定微调收益。
* **日志规范化** 便于对比不同环境/参数的训练效果。
* **快速 100 步只是冒烟**，之后应转向标准检索指标（Recall\@K/MRR）等系统评估。
