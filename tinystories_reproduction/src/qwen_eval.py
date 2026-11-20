import argparse,json,re
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from common import read_jsonl

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct'); args=ap.parse_args(); rows=read_jsonl(args.input)
    tok=AutoTokenizer.from_pretrained(args.model); model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype='auto',device_map='auto'); out=[]
    for r in rows:
        prompt=f'''评价故事续写，只输出 JSON。grammar、creativity、consistency 均为 1 到 5 的整数。格式：{{"grammar":1,"creativity":1,"consistency":1}}\n故事开头：{r["prompt"]}\n续写：{r["completion"]}'''; x=tok(prompt,return_tensors='pt').to(model.device); y=model.generate(**x,max_new_tokens=80,do_sample=False); text=tok.decode(y[0][x['input_ids'].shape[1]:],skip_special_tokens=True); m=re.search(r'\{.*?\}',text,re.S); scores=json.loads(m.group(0)) if m else {'grammar':None,'creativity':None,'consistency':None}; out.append({**r,**scores,'raw_score':text})
    with open(args.output,'w',encoding='utf-8') as f:
        for x in out: f.write(json.dumps(x,ensure_ascii=False)+'\n')
if __name__=='__main__': main()
