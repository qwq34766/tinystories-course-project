$ErrorActionPreference='Stop'; Set-Location (Split-Path $PSScriptRoot -Parent)
foreach($m in @('1m','3m','8m')) { python src/train.py --config "configs/gptneo_$m.yaml" --data data --output "outputs/checkpoints/gptneo_$m" --fraction 1.0 --steps 2000 }
foreach($m in @('1m','3m','8m')) { python src/generate.py --checkpoint "outputs/checkpoints/gptneo_$m/final" --output "outputs/generations/experiment1_$m.jsonl" --temperature 0.5 --num-samples 1 }
