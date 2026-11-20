$ErrorActionPreference='Stop'; Set-Location (Split-Path $PSScriptRoot -Parent)
foreach($t in @(0,0.5,1.0)) { $tag=($t.ToString()).Replace('.','p'); python src/generate.py --checkpoint outputs/checkpoints/gptneo_3m/final --output "outputs/generations/temperature_$tag.jsonl" --temperature $t --num-samples 5 --max-new-tokens 120 }
