# bert
# 0.环境要求
python = 3.11

requirements.txt

# 1. 项目概述
本项目实现了基于BERT (Transformer) 架构的文本分类器，用于区分 20 Newsgroups 数据集中的两个语义接近的类别：alt.atheism（无神论）与 soc.religion.christian（基督教）。

# 2.实验配置
数据集划分,样本数量,占比
训练集 (Train),863,~80% (自原始训练集)
验证集 (Val),216,~20% (自原始训练集)
测试集 (Test),717,独立测试集
