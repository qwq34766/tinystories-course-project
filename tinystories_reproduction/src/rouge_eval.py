import argparse, csv, statistics, random
from pathlib import Path
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from common import repeat_ngram_ratio

def sample_stories(path, n, seed):
    rng=random.Random(seed); sample=[]
    with open(path, encoding='utf-8') as f:
        for i,line in enumerate(f):
            if not line.strip(): continue
            row=__import__('json').loads(line)
            if len(sample)<n: sample.append(row)
            else:
                j=rng.randint(0,i)
                if j<n: sample[j]=row
    return sample

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint-dir',required=True); ap.add_argument('--data',required=True); ap.add_argument('--output',required=True); ap.add_argument('--n',type=int,default=100); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    stories=sample_stories(args.data,args.n,args.seed); scorer=rouge_scorer.RougeScorer(['rouge2'],use_stemmer=False); rows=[]
    for ck in sorted(Path(args.checkpoint_dir).glob('*/final')):
        if ck.parent.name not in {'gptneo_1m','gptneo_3m','gptneo_8m'}: continue
        try: tok=AutoTokenizer.from_pretrained(ck); model=AutoModelForCausalLM.from_pretrained(ck); model.eval(); dev='cuda' if torch.cuda.is_available() else 'cpu'; model.to(dev)
        except Exception: continue
        for row in stories:
            words=row['text'].split(); cut=max(1,int(len(words)*.4)); prompt=' '.join(words[:cut]); ref=' '.join(words[cut:]); x=tok(prompt,return_tensors='pt').to(dev)
            with torch.no_grad(): y=model.generate(**x,max_new_tokens=60,do_sample=False,pad_token_id=tok.eos_token_id)
            gen=tok.decode(y[0][x['input_ids'].shape[1]:],skip_special_tokens=True); s=scorer.score(ref,gen)['rouge2']; overlap=len(set(gen.split())&set(ref.split()))/max(1,len(set(ref.split())))
            rows.append({'model':ck.parent.name,'story_id':row.get('id'),'prompt_length':len(prompt.split()),'reference_length':len(ref.split()),'generation_length':len(gen.split()),'rouge2_precision':s.precision,'rouge2_recall':s.recall,'rouge2_f1':s.fmeasure,'token_overlap':overlap,'repeat_4gram_ratio':repeat_ngram_ratio(gen)})
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['model']); w.writeheader(); w.writerows(rows)
if __name__=='__main__': main()
