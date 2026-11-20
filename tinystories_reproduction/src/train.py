import argparse, json, time
from array import array
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from build_model import build_model, count_parameters
from common import seed_everything, read_jsonl

class StoryDataset(Dataset):
    """Lazy JSONL dataset: stores byte offsets, tokenizes only the requested item."""
    def __init__(self, path, tokenizer, max_length=512, fraction=1.0, seed=42, max_samples=0):
        self.path=path; self.tokenizer=tokenizer; self.max_length=max_length
        self.offsets=array('Q'); pos=0
        with open(path,'rb') as f:
            for line in f:
                if line.strip(): self.offsets.append(pos)
                pos += len(line)
        n=max(1,int(len(self.offsets)*fraction))
        if max_samples: n=min(n,max_samples)
        if n < len(self.offsets):
            g=torch.Generator().manual_seed(seed)
            self.indices=array('Q', torch.randperm(len(self.offsets),generator=g)[:n].tolist())
        else: self.indices=None
    def __len__(self): return len(self.indices) if self.indices is not None else len(self.offsets)
    def __getitem__(self,i):
        source_i=int(self.indices[i]) if self.indices is not None else i
        with open(self.path,'rb') as f:
            f.seek(self.offsets[source_i]); row=json.loads(f.readline().decode('utf-8'))
        x=self.tokenizer(row['text'], truncation=True, max_length=self.max_length, padding='max_length')
        return {k:torch.tensor(v) for k,v in x.items()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--data',default='data'); ap.add_argument('--output',required=True); ap.add_argument('--fraction',type=float,default=1.0); ap.add_argument('--steps',type=int,default=2000); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--max-length',type=int,default=512); ap.add_argument('--max-eval-samples',type=int,default=256); ap.add_argument('--eval-steps',type=int,default=250); ap.add_argument('--resume-from-checkpoint',default=''); args=ap.parse_args()
    seed_everything(args.seed); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M', use_fast=True, local_files_only=True); tokenizer.pad_token=tokenizer.eos_token
    vocab_size=tokenizer.vocab_size; model,cfg=build_model(args.config,vocab_size)
    train=StoryDataset(f'{args.data}/train.jsonl',tokenizer,args.max_length,args.fraction,args.seed); val=StoryDataset(f'{args.data}/validation.jsonl',tokenizer,args.max_length,1.0,args.seed,args.max_eval_samples)
    device='cuda' if torch.cuda.is_available() else 'cpu'; print(f'device={device} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    t0=time.time(); ta=TrainingArguments(output_dir=str(out), max_steps=args.steps, per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8, learning_rate=3e-4, weight_decay=0.1, logging_steps=50, eval_strategy='steps', eval_steps=args.eval_steps, save_steps=args.eval_steps, save_total_limit=2, report_to='none', fp16=torch.cuda.is_available(), dataloader_num_workers=0, gradient_checkpointing=False, seed=args.seed)
    trainer=Trainer(model=model,args=ta,train_dataset=train,eval_dataset=val,data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False))
    resume=args.resume_from_checkpoint or None
    trainer.train(resume_from_checkpoint=resume); trainer.save_model(str(out/'final')); tokenizer.save_pretrained(str(out/'final'))
    json.dump({'config':cfg,'parameters':count_parameters(model),'fraction':args.fraction,'steps':args.steps,'seconds':time.time()-t0},open(out/'metadata.json','w',encoding='utf-8'),indent=2)
if __name__=='__main__': main()
