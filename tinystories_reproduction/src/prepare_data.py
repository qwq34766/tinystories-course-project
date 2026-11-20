import argparse, json, re
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from common import seed_everything, write_jsonl

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='data'); ap.add_argument('--max-vocab-size',type=int,default=10000); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--train-file'); ap.add_argument('--valid-file'); ap.add_argument('--max-stories',type=int,default=0); args=ap.parse_args()
    seed_everything(args.seed); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M', use_fast=True, local_files_only=True)
    if args.train_file and args.valid_file:
        def parse(path):
            rows=[]; buf=[]
            with open(path, encoding='utf-8') as f:
                for line in f:
                    if '<|endoftext|>' in line:
                        before=line.split('<|endoftext|>',1)[0]; buf.append(before)
                        text=''.join(buf).strip();
                        if text: rows.append({'id':len(rows),'text':text})
                        if args.max_stories and len(rows)>=args.max_stories: return rows
                        buf=[]
                    else: buf.append(line)
            text=''.join(buf).strip()
            if text: rows.append({'id':len(rows),'text':text})
            return rows
        local={'train':parse(args.train_file),'validation':parse(args.valid_file)}
    else:
        from datasets import load_dataset
        ds=load_dataset('roneneldan/TinyStories')
        local={s:[{'id':i,'text':r['text']} for i,r in enumerate(ds[s]) if r.get('text','').strip()] for s in ds}
    counter=Counter()
    for row in local['train']:
        counter.update(tok.encode(row['text'], add_special_tokens=False))
    vocab=counter.most_common(args.max_vocab_size)
    json.dump({'tokenizer':'EleutherAI/gpt-neo-125M','max_vocab_size':args.max_vocab_size,'token_ids':[x[0] for x in vocab]},open(out/'vocab.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    for split,rows in local.items(): write_jsonl(out/f'{split}.jsonl',rows)
    print('prepared', {s:len(rows) for s,rows in local.items()}, 'output=',out)
if __name__=='__main__': main()
