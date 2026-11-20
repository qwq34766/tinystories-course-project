$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
foreach ($m in @('1m','3m','8m')) {
    python -u src/qwen_ollama_eval.py --input "outputs/generations/experiment1_$m.jsonl" --output "outputs/metrics/qwen_experiment1_$m.jsonl"
    if ($LASTEXITCODE -ne 0) { throw 'Qwen scoring failed' }
}
foreach ($tag in @('0.01','0.1','1')) {
    python -u src/qwen_ollama_eval.py --input "outputs/generations/data_scale_$tag.jsonl" --output "outputs/metrics/qwen_data_scale_$tag.jsonl"
    if ($LASTEXITCODE -ne 0) { throw 'Qwen scoring failed' }
}
foreach ($tag in @('0','0p5','1p0')) {
    python -u src/qwen_ollama_eval.py --input "outputs/generations/temperature_$tag.jsonl" --output "outputs/metrics/qwen_temperature_$tag.jsonl"
    if ($LASTEXITCODE -ne 0) { throw 'Qwen scoring failed' }
}
