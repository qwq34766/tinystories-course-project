import argparse, json
from pathlib import Path
from experiment_runtime import generate_rows
from common import read_jsonl

PROMPTS=[('factual_01','Tom saw a bird in the tree. He wondered what birds like to eat.'),('reasoning_01','Alice was tired after playing outside. She saw a warm bed and'),('context_01','Lily put her red ball in the box. Tom took the blue car. Then Lily asked Tom where')]
def load_prompts(path=None):
    if path and Path(path).exists(): return [(r['prompt_id'],r['prompt']) for r in read_jsonl(path)]
    templates=[('factual','Tom saw a bird in the tree. He wondered what birds like to eat.'),('factual','Mary found a small fish in the pond. She knew the fish needed water.'),('reasoning','Alice was tired after playing outside. She saw a warm bed and'),('reasoning','The box was too high for Lily to reach. She found a chair and'),('context','Lily put her red ball in the box. Tom took the blue car. Then Lily asked Tom where'),('context','Tom put a book on the desk and Jane put a pencil in the drawer. Later')]
    return [(f'{kind}_{i+1:02d}',text) for i in range(50) for kind,text in [templates[i%len(templates)]]]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True); ap.add_argument('--temperature',type=float,default=.5); ap.add_argument('--num-samples',type=int,default=1); ap.add_argument('--max-new-tokens',type=int,default=120); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--prompts'); args=ap.parse_args()
    prompts=read_jsonl(args.prompts or 'configs/eval_prompts.jsonl')
    generate_rows(args.checkpoint,prompts,args.output,args.temperature,args.num_samples,args.max_new_tokens,args.seed)
if __name__=='__main__': main()
