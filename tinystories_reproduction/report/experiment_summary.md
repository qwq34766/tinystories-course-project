# TinyStories 四项实验结果汇总

生成时间：2026-10-27T13:55:41.125689

## 实验边界

- GPT-Neo 从头训练；正式模型固定 2000 optimizer steps，实际参数量约 1.65M、3.45M、7.29M。
- 1%、10% 为独立训练，100% 组复用已有 3M checkpoint；比较的是固定训练步数下的数据规模影响。
- 本实现使用 GPT-Neo tokenizer 和 512 context，属于论文配置的可复现近似，不能称为原论文完整复现。
- Qwen 分数来自本地 Qwen2.5-1.5B-Instruct GGUF 经 Ollama 的替代评分，不等价于论文 GPT-4 评分。

## 实验二：Rouge-2

- 记录：`outputs\metrics\experiment2_rouge2.csv`，共 300 条（3 个模型 × 100 个固定随机抽样故事）。
- 每个故事前 40% 为 prompt，后 60% 为 reference。

## 改进实验一：数据规模

- 1%、10%、100% 均有生成记录；100% 组明确复用已有 3M checkpoint。
- 三组训练预算均为 2000 steps，不是完整 epoch 对比。

## 改进实验二：temperature

- 0、0.5、1.0 各使用相同 50 个 prompt，每个 prompt 5 条，共 250 条；0 使用 greedy，其他温度使用 top-k=50、top-p=0.95。

## Qwen 替代评分

- `qwen_*.jsonl` 保留原始返回文本和解析错误；`qwen_summary.csv` 汇总成功解析的分数。

## 限制

- 小模型在固定步数下出现重复和语法问题；Rouge-2 低值不能单独解释为记忆性低，应与验证 loss、重复率和生成案例联合分析。
- 10% 组已从头重新训练并覆盖正式目录；此前中断版本已保留为可追溯备份目录。
