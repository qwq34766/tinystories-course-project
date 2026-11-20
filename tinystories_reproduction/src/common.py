from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + '\n')

def repeat_ngram_ratio(text, n=4):
    words=text.split(); grams=[tuple(words[i:i+n]) for i in range(max(0,len(words)-n+1))]
    return 0.0 if not grams else 1-len(set(grams))/len(grams)

def distinct_ratio(text, n=2):
    words=text.split(); grams=[tuple(words[i:i+n]) for i in range(max(0,len(words)-n+1))]
    return 0.0 if not grams else len(set(grams))/len(grams)
