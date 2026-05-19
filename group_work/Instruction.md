算法实现：
综合实现集成一系列加速方法
报告加速/优化的效果和性能提升

我们想要融合的机制：Muti-head latent attention（attention机制）,you only cache once（层间KV压缩）,然后我之前的src/improved里的snapkv_sink_adaptive（逐层kV压缩）。要求是这三个大的层面中的方法各选一个

Baseline设置
所有实验都应该在Pythia-70M模型上用无训练方法进行优化
在pg-19, wikitext等数据集上进行ppl测试和加速测试
注：pg-19为超长文本数据集，可取单一sample进行测试即可

参考加速指标
TTFT: Time To First Token
TPOT: Time Per Output Token
Throughput
Total/average FLOPs: Floating Point Operations over the sequence

论文要求：
使用NeurIPS模板撰写英文论文（在压缩包提供），正文不超过4页
包含abstract, intro, method, experiments
随论文提交可以复现实验结果的代码（可以是链接附在论文内；个人部分的链接也贴在论文里即可）

创新：
设计出相应的改进算法，将根据改进算法的质量和有效性进行评分
可以在某个已有的算法基础上进行改进，若是在已有方法上进行的原创性改进，应在报告中汇报其本身算法的参考文献
