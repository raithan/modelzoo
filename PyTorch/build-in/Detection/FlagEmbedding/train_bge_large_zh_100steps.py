#!/usr/bin/env python3
import os,json,random,argparse
from pathlib import Path
import torch
import torch_sdaa
from torch import nn
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,AutoModel

class TripletDataset(Dataset):
    def __init__(self,path,max_query_len=64,max_passage_len=512,neg_k=1):
        self.data=[]
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                o=json.loads(line)
                if 'query' in o and 'pos' in o and 'neg' in o and o['pos'] and o['neg']:
                    self.data.append(o)
        self.max_query_len=max_query_len
        self.max_passage_len=max_passage_len
        self.neg_k=neg_k
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        r=self.data[i]
        return r['query'],random.choice(r['pos']),random.sample(r['neg'],min(self.neg_k,len(r['neg'])))

def mean_pool(h,m):
    m=m.unsqueeze(-1).type_as(h)
    return (h*m).sum(1)/m.sum(1).clamp(min=1e-6)

def build_loader(path,tokenizer,batch_size,mql,mpl,neg_k):
    ds=TripletDataset(path,mql,mpl,neg_k)
    def collate(batch):
        qs,ps,nlists=zip(*batch)
        negs=[n for nl in nlists for n in nl]
        q_tok=tokenizer(list(qs),padding=True,truncation=True,max_length=mql,return_tensors='pt')
        p_tok=tokenizer(list(ps),padding=True,truncation=True,max_length=mpl,return_tensors='pt')
        n_tok=tokenizer(negs,padding=True,truncation=True,max_length=mpl,return_tensors='pt')
        offs=[];c=0
        for nl in nlists:
            offs.append((c,c+len(nl)));c+=len(nl)
        return q_tok,p_tok,n_tok,offs
    return DataLoader(ds,batch_size=batch_size,shuffle=True,drop_last=True,collate_fn=collate)

def contrastive_loss(q,p,n,offs,temp,in_batch=True):
    B=q.size(0)
    if in_batch:
        sim_pp=(q@p.t())/temp
        labels=torch.arange(B,device=q.device)
        rows=[]
        for i in range(B):
            st,ed=offs[i]
            extra=(q[i:i+1]@n[st:ed].t())/temp
            rows.append(torch.cat([sim_pp[i],extra.squeeze(0)],0).unsqueeze(0))
        logits=torch.cat(rows,0)
        return nn.CrossEntropyLoss()(logits,labels)
    else:
        losses=[]
        for i in range(B):
            st,ed=offs[i]
            pos=(q[i]*p[i]).sum().unsqueeze(0)/temp
            negs=(q[i:i+1]@n[st:ed].t()).squeeze(0)/temp
            logits=torch.cat([pos,negs]).unsqueeze(0)
            losses.append(nn.CrossEntropyLoss()(logits,torch.zeros(1,dtype=torch.long,device=q.device)))
        return torch.stack(losses).mean()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model_dir',required=True)
    ap.add_argument('--data_path',required=True)
    ap.add_argument('--steps',type=int,default=100)
    ap.add_argument('--batch_size',type=int,default=4)
    ap.add_argument('--max_query_len',type=int,default=64)
    ap.add_argument('--max_passage_len',type=int,default=512)
    ap.add_argument('--lr',type=float,default=5e-5)
    ap.add_argument('--temperature',type=float,default=0.05)
    ap.add_argument('--neg_each',type=int,default=1)
    ap.add_argument('--log_interval',type=int,default=10)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--fp16',action='store_true')
    ap.add_argument('--no_in_batch_neg',action='store_true')
    args=ap.parse_args()
    random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device('sdaa' if torch.sdaa.is_available() else 'cpu')
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    model=AutoModel.from_pretrained(args.model_dir,local_files_only=True).to(device).train()
    loader=build_loader(args.data_path,tok,args.batch_size,args.max_query_len,args.max_passage_len,args.neg_each)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    scaler=torch.sdaa.amp.GradScaler(enabled=args.fp16 and device.type=='sdaa')
    log_path=Path('sdaa_loss.log')
    if log_path.exists(): log_path.unlink()
    step=0;epoch=1;it=iter(loader)
    with open(log_path,'a',encoding='utf-8') as flog:
        while step<args.steps:
            try: batch=next(it)
            except StopIteration:
                it=iter(loader);batch=next(it);epoch+=1
            q_tok,p_tok,n_tok,offs=batch
            q_tok={k:v.to(device) for k,v in q_tok.items()}
            p_tok={k:v.to(device) for k,v in p_tok.items()}
            n_tok={k:v.to(device) for k,v in n_tok.items()}
            with torch.sdaa.amp.autocast(enabled=args.fp16 and device.type=='sdaa'):
                q_out=model(**q_tok,return_dict=True)
                p_out=model(**p_tok,return_dict=True)
                n_out=model(**n_tok,return_dict=True)
                q_emb=nn.functional.normalize(mean_pool(q_out.last_hidden_state,q_tok['attention_mask']),p=2,dim=-1)
                p_emb=nn.functional.normalize(mean_pool(p_out.last_hidden_state,p_tok['attention_mask']),p=2,dim=-1)
                n_emb=nn.functional.normalize(mean_pool(n_out.last_hidden_state,n_tok['attention_mask']),p=2,dim=-1)
                loss=contrastive_loss(q_emb,p_emb,n_emb,offs,args.temperature,not args.no_in_batch_neg)
            opt.zero_grad()
            if args.fp16 and device.type=='sdaa':
                scaler.scale(loss).backward();scaler.step(opt);scaler.update()
            else:
                loss.backward();opt.step()
            step+=1
            if step%args.log_interval==0 or step==args.steps:
                line=f"[epoch {epoch}/1 | step {step}/{args.steps}] loss {loss.item():.4f}"
                print(line,flush=True);flog.write(line+"\n");flog.flush()
    torch.save(model.state_dict(),Path('final_state.pt'))

if __name__=='__main__':
    main()
