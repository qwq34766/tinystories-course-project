import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib as mpl
mpl.rcParams.update({'font.family':'sans-serif','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
import matplotlib.pyplot as plt

def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=220,bbox_inches='tight'); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--metrics',default='outputs/metrics'); p.add_argument('--figures',default='outputs/figures'); a=p.parse_args(); m=Path(a.metrics); f=Path(a.figures); f.mkdir(parents=True,exist_ok=True)
    rouge=m/'experiment2_rouge2.csv'
    if rouge.exists():
        d=pd.read_csv(rouge); cols=['rouge2_precision','rouge2_recall','rouge2_f1','token_overlap','repeat_4gram_ratio']; d.groupby('model')[cols].agg(['mean','median','count']).to_csv(m/'experiment2_rouge2_summary.csv')
        fig,ax=plt.subplots(figsize=(5,3)); d.boxplot(column='rouge2_precision',by='model',ax=ax); ax.set_title('Rouge-2 precision by model'); ax.set_xlabel(''); ax.set_ylabel('Precision'); fig.suptitle(''); save(fig,f/'experiment2_rouge2_boxplot.png')
        fig,ax=plt.subplots(figsize=(5,3)); [ax.hist(g.rouge2_precision,bins=12,alpha=.55,label=k) for k,g in d.groupby('model')]; ax.set_xlabel('Rouge-2 precision'); ax.set_ylabel('Stories'); ax.legend(); save(fig,f/'experiment2_rouge2_histogram.png')
    scale=[]
    for tag,label in [('0.01','1%'),('0.1','10%'),('1','100%')]:
        q=Path('outputs/generations')/f'data_scale_{tag}.jsonl'
        if q.exists(): x=pd.read_json(q,lines=True); x['data_scale']=label; scale.append(x)
    if scale:
        d=pd.concat(scale,ignore_index=True); cols=['length','rouge2_precision','rouge2_recall','rouge2_f1','distinct_2','distinct_4','repeat_4gram_ratio','token_overlap']; d.to_csv(m/'improvement1_data_scale.csv',index=False); d.groupby('data_scale')[cols].agg(['mean','median','count']).to_csv(m/'improvement1_data_scale_summary.csv')
        for metric,name in [('rouge2_precision','improvement1_rouge2.png'),('distinct_2','improvement1_quality.png')]:
            fig,ax=plt.subplots(figsize=(5,3)); d.boxplot(column=metric,by='data_scale',ax=ax); ax.set_title(metric+' by data scale'); ax.set_xlabel(''); fig.suptitle(''); save(fig,f/name)
    temps=[]
    for tag,label in [('0','0'),('0p5','0.5'),('1p0','1.0')]:
        q=Path('outputs/generations')/f'temperature_{tag}.jsonl'
        if q.exists(): x=pd.read_json(q,lines=True); x['temperature_label']=label; temps.append(x)
    if temps:
        d=pd.concat(temps,ignore_index=True); cols=['length','distinct_2','distinct_4','repeat_4gram_ratio','repeat_sentence_ratio','rouge2_f1','early_termination']; d.to_csv(m/'improvement2_temperature.csv',index=False); d.groupby('temperature_label')[cols].agg(['mean','median','count']).to_csv(m/'improvement2_temperature_summary.csv')
        fig,ax=plt.subplots(figsize=(5,3)); d.groupby('temperature_label')[['distinct_2','distinct_4']].mean().plot(kind='bar',ax=ax); ax.set_ylabel('Distinct ratio'); ax.set_xlabel('Temperature'); save(fig,f/'improvement2_diversity.png')
        fig,ax=plt.subplots(figsize=(5,3)); d.groupby('temperature_label')[['rouge2_f1','repeat_4gram_ratio']].mean().plot(kind='bar',ax=ax); ax.set_ylabel('Mean metric'); ax.set_xlabel('Temperature'); save(fig,f/'improvement2_quality.png')
    qrows=[]
    for q in m.glob('qwen_*.jsonl'):
        if 'previous' in q.name or 'failed' in q.name: continue
        x=pd.read_json(q,lines=True); x['source_label']=q.stem.replace('qwen_',''); qrows.append(x)
    if qrows:
        q=pd.concat(qrows,ignore_index=True); q.groupby('source_label')[['grammar','creativity','consistency']].agg(['mean','count']).to_csv(m/'qwen_summary.csv')
    lines=['# TinyStories 四项实验结果汇总','',f'生成时间：{pd.Timestamp.now().isoformat()}','', '## 实验边界','', '- GPT-Neo 从头训练；正式模型固定 2000 optimizer steps，实际参数量约 1.65M、3.45M、7.29M。','- 1%、10% 为独立训练，100% 组复用已有 3M checkpoint；比较的是固定训练步数下的数据规模影响。','- 本实现使用 GPT-Neo tokenizer 和 512 context，属于论文配置的可复现近似，不能称为原论文完整复现。','- Qwen 分数来自本地 Qwen2.5-1.5B-Instruct GGUF 经 Ollama 的替代评分，不等价于论文 GPT-4 评分。','']
    if rouge.exists(): lines += ['## 实验二：Rouge-2','',f'- 记录：`{rouge}`，共 {len(pd.read_csv(rouge))} 条（3 个模型 × 100 个固定随机抽样故事）。','- 每个故事前 40% 为 prompt，后 60% 为 reference。','']
    if scale: lines += ['## 改进实验一：数据规模','', '- 1%、10%、100% 均有生成记录；100% 组明确复用已有 3M checkpoint。','- 三组训练预算均为 2000 steps，不是完整 epoch 对比。','']
    if temps: lines += ['## 改进实验二：temperature','', '- 0、0.5、1.0 各使用相同 50 个 prompt，每个 prompt 5 条，共 250 条；0 使用 greedy，其他温度使用 top-k=50、top-p=0.95。','']
    if qrows: lines += ['## Qwen 替代评分','', '- `qwen_*.jsonl` 保留原始返回文本和解析错误；`qwen_summary.csv` 汇总成功解析的分数。','']
    lines += ['## 限制','', '- 小模型在固定步数下出现重复和语法问题；Rouge-2 低值不能单独解释为记忆性低，应与验证 loss、重复率和生成案例联合分析。','- 10% 组已从头重新训练并覆盖正式目录；此前中断版本已保留为可追溯备份目录。']
    Path('report').mkdir(exist_ok=True); Path('report/experiment_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__': main()
