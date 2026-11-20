from pathlib import Path
import yaml
from transformers import GPTNeoConfig, GPTNeoForCausalLM

def load_config(path):
    with open(path, encoding='utf-8') as f: return yaml.safe_load(f)

def build_model(config_path, vocab_size, tokenizer_name='EleutherAI/gpt-neo-125M'):
    c=load_config(config_path)
    cfg=GPTNeoConfig(vocab_size=vocab_size, max_position_embeddings=c['max_position_embeddings'],
        hidden_size=c['hidden_size'], num_layers=c['num_layers'], num_heads=c['num_heads'],
        intermediate_size=c['intermediate_size'], attention_types=[[['global'], c['num_layers']]],
        window_size=c['attention_window'], bos_token_id=0, eos_token_id=0)
    return GPTNeoForCausalLM(cfg), c

def count_parameters(model): return sum(p.numel() for p in model.parameters())
