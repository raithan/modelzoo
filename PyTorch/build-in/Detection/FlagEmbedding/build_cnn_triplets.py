#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert CNN/DailyMail .bin chunks (Pointer‑Generator format) to
(query, positive, negative) triples for FlagEmbedding fine‑tuning.

Usage:
  python build_cnn_triplets.py \
      --chunk_dir /data/teco-data/CNN/Daily-Mail/finished_files/chunked \
      --out_path  /root/flag_data/cnn_triplets.jsonl \
      --max_samples 100000          # 可调，None=全量
"""
import os, struct, argparse, json, random, glob
from tqdm import tqdm
import tensorflow as tf

def yield_examples(bin_path):
    with open(bin_path, "rb") as f:
        while True:
            len_bytes = f.read(8)
            if not len_bytes: break
            str_len = struct.unpack("q", len_bytes)[0]
            example_str = f.read(str_len)
            ex = tf.train.Example()
            ex.ParseFromString(example_str)
            art  = ex.features.feature["article"].bytes_list.value[0].decode()
            summ = ex.features.feature["abstract"].bytes_list.value[0].decode()
            yield art, summ

def clean_summary(text):
    # 去掉 <s> ... </s> 标记并合并成一句
    return " ".join(t for t in text.replace("<s>","").replace("</s>","").splitlines() if t.strip())

def main(args):
    chunks = sorted(glob.glob(os.path.join(args.chunk_dir, "*.bin")))
    cache_articles = []
    total_written = 0
    random.seed(42)

    with open(args.out_path, "w", encoding="utf-8") as out:
        for bin_file in tqdm(chunks, desc="processing chunks"):
            for art, summ in yield_examples(bin_file):
                summ = clean_summary(summ)
                if not summ or not art: continue
                cache_articles.append(art)
                if len(cache_articles) < 2:   # 先累积
                    continue
                neg_art = random.choice(cache_articles[:-1])
                triplet = {"query": summ, "positive": art, "negative": neg_art}
                out.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                total_written += 1
                if args.max_samples and total_written >= args.max_samples:
                    print(f"Reached {total_written}; early stop.")
                    return
    print(f"Finished: {total_written} triplets written to {args.out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chunk_dir", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--max_samples", type=int, default=None)
    main(p.parse_args())
