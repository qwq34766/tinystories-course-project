$ErrorActionPreference='Stop'; Set-Location (Split-Path $PSScriptRoot -Parent)
foreach($f in @('0.01','0.1')) { $tag=$f.Replace('.','p'); python src/train.py --config configs/gptneo_3m.yaml --data data --output "outputs/checkpoints/gptneo_3m_fraction_$tag" --fraction $f --steps 2000 --max-eval-samples 256 --eval-steps 250 }
