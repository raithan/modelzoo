#!/usr/bin/env python3
import torch
import torch_sdaa
import os, json, math, argparse, random, time
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

class TripletDataset(Dataset):
    def __init__(self, jsonl_path, max_query_len=64, max_passage_len=256, sample_neg_k=1):
        self.items = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                if "query" not in obj or "pos" not in obj or "neg" not in obj:
                    continue
                if not obj["pos"] or not obj["neg"]:
                    continue
                self.items.append({
                    "query": obj["query"],
                    "pos": obj["pos"],
                    "neg": obj["neg"]
                })
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len
        self.sample_neg_k = sample_neg_k
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        r = self.items[idx]
        pos_text = random.choice(r["pos"])
        neg_list = random.sample(r["neg"], k=min(self.sample_neg_k, len(r["neg"])))
        return r["query"], pos_text, neg_list

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts

def build_dataloader(path, tokenizer, batch_size, max_query_len, max_passage_len, sample_neg_k):
    ds = TripletDataset(path, max_query_len, max_passage_len, sample_neg_k)
    def collate(batch):
        queries, poss, neg_lists = zip(*batch)
        flat_negs = [n for negs in neg_lists for n in negs]
        q_tok = tokenizer(list(queries), padding=True, truncation=True,
                          max_length=max_query_len, return_tensors='pt')
        p_tok = tokenizer(list(poss), padding=True, truncation=True,
                          max_length=max_passage_len, return_tensors='pt')
        n_tok = tokenizer(flat_negs, padding=True, truncation=True,
                          max_length=max_passage_len, return_tensors='pt')
        neg_offsets = []
        c = 0
        for negs in neg_lists:
            neg_offsets.append((c, c + len(negs)))
            c += len(negs)
        return q_tok, p_tok, n_tok, neg_offsets
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      drop_last=True, collate_fn=collate)

def contrastive_loss(query_emb, pos_emb, neg_emb, neg_offsets, temperature, in_batch_neg=True):
    # query_emb: [B, D]; pos_emb: [B, D]; neg_emb: [Nneg, D]
    B = query_emb.size(0)
    if in_batch_neg:
        # Similarity matrix with all positives (in-batch).
        sim_pp = torch.matmul(query_emb, pos_emb.t()) / temperature  # [B,B]
        labels = torch.arange(B, device=query_emb.device)
        # Append explicit negatives per row.
        logits_list = []
        for i in range(B):
            row = sim_pp[i]  # [B]
            start, end = neg_offsets[i]
            extra = torch.matmul(query_emb[i:i+1], neg_emb[start:end].t()) / temperature  # [1, k]
            row = torch.cat([row, extra.squeeze(0)], dim=0)
            logits_list.append(row.unsqueeze(0))
        logits = torch.cat(logits_list, dim=0)  # [B, B+k_i]
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss
    else:
        # Only own pos + its negatives
        losses = []
        for i in range(B):
            start, end = neg_offsets[i]
            pos_sim = (query_emb[i] * pos_emb[i]).sum(0, keepdim=True) / temperature
            neg_sims = torch.matmul(query_emb[i:i+1], neg_emb[start:end].t()).squeeze(0) / temperature
            all_logits = torch.cat([pos_sim, neg_sims], dim=0).unsqueeze(0)  # [1,1+k]
            labels = torch.zeros(1, dtype=torch.long, device=query_emb.device)
            losses.append(nn.CrossEntropyLoss()(all_logits, labels))
        return torch.stack(losses).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="./outputs/bge_ft_cnn_custom")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_group_size", type=int, default=8)  # (1 pos + k neg) target size for difficulty
    parser.add_argument("--max_query_len", type=int, default=64)
    parser.add_argument("--max_passage_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--neg_each", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_in_batch_neg", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = Path("sdaa_loss.log")
    if log_path.exists():
        log_path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "sdaa" if torch.cuda.device_count()==0 else "cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True)
    model.to(device)
    model.train()

    dataloader = build_dataloader(args.data_path, tokenizer, args.batch_size,
                                  args.max_query_len, args.max_passage_len, args.neg_each)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    step = 0
    epoch = 1
    data_iter = iter(dataloader)

    with open(log_path, "a", encoding="utf-8") as flog:
        while step < args.steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
                epoch += 1

            q_tok, p_tok, n_tok, neg_offsets = batch
            q_tok = {k: v.to(device) for k, v in q_tok.items()}
            p_tok = {k: v.to(device) for k, v in p_tok.items()}
            n_tok = {k: v.to(device) for k, v in n_tok.items()}

            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                q_out = model(**q_tok, return_dict=True)
                p_out = model(**p_tok, return_dict=True)
                n_out = model(**n_tok, return_dict=True)

                q_emb = mean_pool(q_out.last_hidden_state, q_tok["attention_mask"])
                p_emb = mean_pool(p_out.last_hidden_state, p_tok["attention_mask"])
                n_emb = mean_pool(n_out.last_hidden_state, n_tok["attention_mask"])

                q_emb = nn.functional.normalize(q_emb, p=2, dim=-1)
                p_emb = nn.functional.normalize(p_emb, p=2, dim=-1)
                n_emb = nn.functional.normalize(n_emb, p=2, dim=-1)

                loss = contrastive_loss(
                    q_emb, p_emb, n_emb, neg_offsets,
                    temperature=args.temperature,
                    in_batch_neg=not args.no_in_batch_neg
                )

            optimizer.zero_grad()
            if args.fp16 and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            step += 1
            if step % args.log_interval == 0 or step == args.steps:
                line = f"[epoch {epoch}/1 | step {step}/{args.steps}] loss {loss.item():.4f}"
                print(line, flush=True)
                flog.write(line + "\n")
                flog.flush()

    # 可选保存最终模型
    torch.save(model.state_dict(), Path(args.output_dir) / "final_state.pt")

if __name__ == "__main__":
    # 离线执行可提前 export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    main()
