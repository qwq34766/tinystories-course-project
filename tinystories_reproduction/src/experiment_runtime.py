"""Shared deterministic sampling, batched generation and metrics."""
import hashlib, json, random, re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from common import seed_everything, repeat_ngram_ratio, distinct_ratio, read_jsonl

SCORER = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=False)

def fixed_sample(path, n=100, seed=42):
    rng=random.Random(seed); sample=[]; count=0
    with open(path, encoding='utf-8') as f:
        for line_index,line in enumerate(f):
            if not line.strip(): continue
            if count<n: sample.append((line_index,line))
            else:
                j=rng.randrange(count+1)
                if j<n: sample[j]=(line_index,line)
            count+=1
    result=[]
    for index,line in sorted(sample):
        row=json.loads(line); words=row['text'].split(); cut=max(1,int(len(words)*.4))
        result.append({'story_id':row.get('id',index),'source_line':index,'source_file':str(path),
                       'prompt_id':f'story_{index}','prompt':' '.join(words[:cut]),
                       'reference':' '.join(words[cut:]),'split_fraction_words':.4,'selection_seed':seed})
    if len(result)!=n or any(not r['reference'] for r in result): raise ValueError('Insufficient stories')
    return result

def text_metrics(text,reference=''):
    sentences=[s.strip() for s in re.split(r'[.!?]+',text) if s.strip()]
    s=SCORER.score(reference,text)['rouge2']
    return {'length':len(text.split()),'distinct_2':distinct_ratio(text,2),'distinct_4':distinct_ratio(text,4),
            'repeat_4gram_ratio':repeat_ngram_ratio(text),
            'repeat_sentence_ratio':0 if not sentences else 1-len(set(sentences))/len(sentences),
            'rouge2_precision':s.precision,'rouge2_recall':s.recall,'rouge2_f1':s.fmeasure,
            'token_overlap':len(set(text.split())&set(reference.split()))/max(1,len(set(reference.split())))}

def generate_rows(checkpoint,prompts,output,temperature=.5,samples=1,max_new_tokens=120,seed=42,batch_size=10):
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    old=read_jsonl(output) if output.exists() else []
    done={(r['prompt_id'],r['sample_id']) for r in old}
    tok=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,padding_side='left'); tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True).eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    max_prompt=model.config.max_position_embeddings-max_new_tokens
    config={'checkpoint':str(Path(checkpoint).resolve()),'temperature':temperature,'samples':samples,
            'max_new_tokens':max_new_tokens,'seed':seed,'batch_size':batch_size,'top_k':50,'top_p':.95,
            'prompts_sha256':hashlib.sha256(json.dumps(prompts,sort_keys=True).encode()).hexdigest(),
            'device':str(model.device),'max_prompt_tokens':max_prompt}
    cp=output.with_suffix('.config.json')
    if cp.exists() and json.loads(cp.read_text(encoding='utf-8'))!=config: raise ValueError(f'Config changed: {output}')
    cp.write_text(json.dumps(config,indent=2),encoding='utf-8')
    rows=list(old)
    with open(output,'a',encoding='utf-8') as f:
        for sample in range(samples):
            for start in range(0,len(prompts),batch_size):
                batch=prompts[start:start+batch_size]
                if all((p['prompt_id'],sample) in done for p in batch): continue
                batch_seed=seed+sample*10000+start; seed_everything(batch_seed)
                x=tok([p['prompt'] for p in batch],padding=True,truncation=True,max_length=max_prompt,return_tensors='pt').to(model.device)
                kw=dict(max_new_tokens=max_new_tokens,do_sample=temperature>0,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
                if temperature>0: kw.update(temperature=temperature,top_k=50,top_p=.95)
                with torch.inference_mode(): ys=model.generate(**x,**kw)
                for p,y in zip(batch,ys):
                    if (p['prompt_id'],sample) in done: continue
                    tokens=y[x.input_ids.shape[1]:].tolist(); ended=tok.eos_token_id in tokens
                    num=tokens.index(tok.eos_token_id)+1 if ended else len(tokens)
                    text=tok.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    row={**p,'model':Path(checkpoint).parent.name,'checkpoint':str(Path(checkpoint).resolve()),
                         'sample_id':sample,'temperature':temperature,'seed':batch_seed,'batch_start':start,
                         'generation_tokens':num,'early_termination':ended and num<max_new_tokens,
                         'prompt_token_length':min(len(tok(p['prompt']).input_ids),max_prompt),
                         'prompt_truncated':len(tok(p['prompt']).input_ids)>max_prompt,
                         'completion':text,**text_metrics(text,p.get('reference',''))}
                    f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush(); rows.append(row); done.add((p['prompt_id'],sample))
                print(f'{output.name}: {len(rows)}/{len(prompts)*samples}',flush=True)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return rows
