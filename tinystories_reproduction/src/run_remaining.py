"""Run all remaining experiment generation; checkpoints are never retrained here."""
import argparse, csv, json, platform, importlib.metadata
from pathlib import Path
import torch
from common import read_jsonl, write_jsonl
from experiment_runtime import fixed_sample, generate_rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['rouge','scale','temperature','all'],default='all'); a=p.parse_args()
    Path('outputs/metrics').mkdir(parents=True,exist_ok=True)
    if a.stage in ('all','rouge'):
        sample_path=Path('configs/rouge_stories.jsonl')
        if not sample_path.exists(): write_jsonl(sample_path,fixed_sample('data/train.jsonl',100,42))
        prompts=read_jsonl(sample_path); records=[]
        for name in ['gptneo_1m','gptneo_3m','gptneo_8m']:
            rows=generate_rows(f'outputs/checkpoints/{name}/final',prompts,
                               f'outputs/generations/experiment2_{name}.jsonl',0,1,120)
            for r in rows:
                records.append({'model':name,'checkpoint':r['checkpoint'],'story_id':r['story_id'],
                                'prompt_length':len(r['prompt'].split()),'reference_length':len(r['reference'].split()),
                                'generation_length':r['length'],**{k:r[k] for k in
                                ['rouge2_precision','rouge2_recall','rouge2_f1','token_overlap','repeat_4gram_ratio']}})
        with open('outputs/metrics/experiment2_rouge2.csv','w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    if a.stage in ('all','scale'):
        path=Path('configs/data_scale_prompts.jsonl')
        if not path.exists(): write_jsonl(path,fixed_sample('data/validation.jsonl',50,42))
        prompts=read_jsonl(path)
        for fraction,name in [(.01,'gptneo_3m_fraction_0p01'),(.1,'gptneo_3m_fraction_0p1'),(1.,'gptneo_3m')]:
            generate_rows(f'outputs/checkpoints/{name}/final',prompts,
                          f'outputs/generations/data_scale_{fraction:g}.jsonl',.5,1,120)
    if a.stage in ('all','temperature'):
        prompts=read_jsonl('configs/eval_prompts.jsonl')
        if len(prompts)!=50 or len({r['prompt'] for r in prompts})!=50: raise ValueError('Need 50 unique prompts')
        for t,tag in [(0,'0'),(.5,'0p5'),(1.,'1p0')]:
            generate_rows('outputs/checkpoints/gptneo_3m/final',prompts,
                          f'outputs/generations/temperature_{tag}.jsonl',t,5,120)
    env={'python':platform.python_version(),'platform':platform.platform(),
         'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','datasets','rouge-score','pandas','matplotlib']}}
    Path('outputs/metrics/environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')

if __name__=='__main__': main()
