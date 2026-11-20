# TinyStories 四项实验

本项目使用 Hugging Face `GPTNeoConfig`/`GPTNeoForCausalLM` 从头训练小型 GPT-Neo，在 TinyStories 上完成：

1. 1M/3M/8M 模型故事续写对比；
2. Figure 14 风格的生成续写 Rouge-2 分析；
3. 1%、10%、100% 数据量对比；
4. temperature=0/0.5/1.0 的质量与多样性对比。

训练模型不是 Qwen。Qwen2.5 只作为可选本地评分器；没有 Qwen 时，规则指标仍可运行。

## 环境

建议 Python 3.10/3.11。Windows 下不需要 bitsandbytes。

```powershell
conda create -n tinystories python=3.11 -y
conda activate tinystories
pip install -r requirements.txt
```

预处理采用 GPT-2/GPT-Neo tokenizer，并按训练集频率建立 10,000 个 token 的压缩词表；这比直接把原 tokenizer 的前 10,000 个 ID 截断更安全。原始论文的精确 tokenizer 训练版本未随论文完全公开，因此该处属于可审计的实现近似。

## 快速验证

```powershell
.\scripts\run_smoke_test.ps1
```

## 正式流程

```powershell
python src/prepare_data.py --output data --max-vocab-size 10000
.\scripts\run_experiment1.ps1
python src/generate.py --checkpoint outputs/checkpoints/gptneo_3m --output outputs/generations/experiment1.jsonl --temperature 0.5
python src/rouge_eval.py --checkpoint-dir outputs/checkpoints --data data/train.jsonl --output outputs/metrics/experiment2_rouge2.csv
.\scripts\run_improvement1.ps1
.\scripts\run_improvement2.ps1
python src/summarize_results.py --metrics outputs/metrics --figures outputs/figures
```

完整训练可能需要较长时间。先用 smoke test 检查 tokenizer、模型 forward、单步训练和生成，再扩大数据规模。

## 评分边界

`qwen_eval.py` 使用本地 Qwen2.5-Instruct 时，结果是替代评分，不是论文 GPT-4 分数。报告应同时保留验证 loss、Rouge-2、distinct-2/4、重复率和人工抽样结果。
