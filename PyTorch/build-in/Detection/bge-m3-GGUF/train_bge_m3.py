# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAM
#!/usr/bin/env python3
import json, os, math, argparse, random, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# =============== Dataset ===============
class TripletDataset(Dataset):
    def __init__(self, jsonl_path, max_query_len=64, max_passage_len=512, neg_each=1):
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                o = json.loads(line)
                if 'query' in o and 'pos' in o and 'neg' in o:
                    # 只用第一条正例；负例取前 neg_each 条
                    self.samples.append((
                        o['query'],
                        o['pos'][0],
                        o['neg'][:neg_each] if isinstance(o['neg'], list) else [o['neg']]
                    ))
        self.q_len = max_query_len
        self.p_len = max_passage_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q, p, negs = self.samples[idx]
        return q, p, negs

# =============== Utils ===============
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

def collate_fn(batch, tokenizer, q_max, p_max):
    qs, ps, neg_lists = zip(*batch)
    flat_negs = [n for nl in neg_lists for n in nl]

    q_tok = tokenizer(list(qs), padding=True, truncation=True,
                      max_length=q_max, return_tensors='pt')
    p_tok = tokenizer(list(ps), padding=True, truncation=True,
                      max_length=p_max, return_tensors='pt')
    n_tok = tokenizer(flat_negs, padding=True, truncation=True,
                      max_length=p_max, return_tensors='pt')

    # 记录每个样本在 flat_negs 里的区间
    offs = []
    c = 0
    for nl in neg_lists:
        offs.append((c, c+len(nl)))
        c += len(nl)
    return q_tok, p_tok, n_tok, offs

def info_nce_loss(q_emb, p_emb, n_emb, offs, temperature=0.15):
    """
    对每条样本：
      正例得分 = q_emb[i] · p_emb[i]
      负例得分 = q_emb[i] 与对应显式 neg 的相似度
      还包含 in-batch negatives（其它样本的 p_emb）
    最终拼成一行 logits，目标 label 就是自己的正例索引。
    """
    B = q_emb.size(0)
    # in-batch 部分 (B,B)
    in_batch_logits = (q_emb @ p_emb.T) / temperature
    rows = []
    for i in range(B):
        st, ed = offs[i]
        explicit_neg_logits = (q_emb[i:i+1] @ n_emb[st:ed].T) / temperature  # (1, neg_each)
        row = torch.cat([in_batch_logits[i], explicit_neg_logits.squeeze(0)], dim=0)
        rows.append(row.unsqueeze(0))
    logits = torch.cat(rows, dim=0)  # (B, B + neg_each)
    labels = torch.arange(B, device=logits.device)
    return nn.CrossEntropyLoss()(logits, labels)

# =============== Main ===============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--neg_each', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--temperature', type=float, default=0.15)
    parser.add_argument('--q_max_len', type=int, default=64)
    parser.add_argument('--p_max_len', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='outputs/bge_m3_ft')
    parser.add_argument('--save_every', type=int, default=0, help='0=only final, >0 save every N steps')
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True)

    device = torch.device('sdaa' if hasattr(torch.backends, 'sdaa') else 'cpu')
    model.to(device)
    model.train()

    # Dataset / Loader
    ds = TripletDataset(args.data_path, args.q_max_len, args.p_max_len, args.neg_each)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.q_max_len, args.p_max_len),
        drop_last=True
    )

    total_steps = args.steps
    epochs = math.ceil(total_steps / len(dl))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=0, num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    step = 0
    for epoch in range(1, epochs+1):
        for batch in dl:
            if step >= total_steps:
                break
            q_tok, p_tok, n_tok, offs = batch
            for d in (q_tok, p_tok, n_tok):
                d.to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                q_out = model(**q_tok, return_dict=True)
                p_out = model(**p_tok, return_dict=True)
                n_out = model(**n_tok, return_dict=True)

                q_emb = F.normalize(mean_pool(q_out.last_hidden_state, q_tok['attention_mask']), p=2, dim=-1)
                p_emb = F.normalize(mean_pool(p_out.last_hidden_state, p_tok['attention_mask']), p=2, dim=-1)
                n_emb = F.normalize(mean_pool(n_out.last_hidden_state, n_tok['attention_mask']), p=2, dim=-1)

                loss = info_nce_loss(q_emb, p_emb, n_emb, offs, temperature=args.temperature)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            scheduler.step()

            step += 1
            if step % args.log_interval == 0 or step == total_steps:
                print(f"[epoch {epoch}/1 | step {step}/{total_steps}] loss {loss.item():.4f}")

            if args.save_every > 0 and step % args.save_every == 0:
                save_path = Path(args.save_dir) / f"step_{step}"
                save_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

        if step >= total_steps:
            break

    # final save
    final_dir = Path(args.save_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Training done. Saved to: {final_dir}")

if __name__ == "__main__":
    main()
