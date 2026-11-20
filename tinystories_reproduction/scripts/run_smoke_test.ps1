$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
python src/prepare_data.py --output data_smoke --max-vocab-size 10000 --max-stories 2000 --train-file ..\TinyStoriesV2-GPT4-train.txt --valid-file ..\TinyStoriesV2-GPT4-valid.txt
python src/train.py --config configs/gptneo_1m.yaml --data data_smoke --output outputs/checkpoints/smoke_1m --fraction 0.0005 --steps 2 --max-length 128 --max-eval-samples 32 --eval-steps 2
python src/generate.py --checkpoint outputs/checkpoints/smoke_1m/final --output outputs/generations/smoke.jsonl --temperature 0.5 --max-new-tokens 40
Write-Host 'Smoke test complete.'
