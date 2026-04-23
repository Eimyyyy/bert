# bert
# 0.环境要求
python = 3.11

requirements.txt

# 1. 项目概述
本项目实现了基于BERT (Transformer) 架构的文本分类器，用于区分 20 Newsgroups 数据集中的两个语义接近的类别：alt.atheism（无神论）与 soc.religion.christian（基督教）。

# 2.实验配置
## 2.1 数据划分
- **原始训练集大小**：1079
- **拆分后训练集**：863 (80%)
- **验证集**：216 (20%)
- **测试集**：717
## 2.2 架构参数
| 参数名称 | 数值 | 说明 |
| :--- | :--- | :--- |
| **vocab_size** | 动态 | 由预处理生成的 `word_to_idx` 词典长度决定 |
| **max_len** | **256** | 最大输入序列长度（超过截断，不足补 0） |
| **hidden_size** | **256** | 词嵌入维度及全连接层宽度 |
| **n_layers** | **6** | Transformer Encoder Block 的堆叠层数 |
| **attn_heads** | **8** | 多头自注意力机制的头数 ($d_k = 32$) |
| **dropout** | **0.1** | Transformer 内部层、注意力权重、嵌入层的丢弃率 |
## 2.3 超参数
| 参数名称 | 数值 | 说明 |
| :--- | :--- | :--- |
| **batch_size** | **16** | 训练批次大小 |
| **learning_rate** | **1e-4** | 初始学习率 (Adam Optimizer) |
| **max_epochs** | **100** | 最大迭代轮数（配合早停机制使用） |
| **loss_func** | `CrossEntropyLoss` | 交叉熵损失函数 |
| **device** | `cuda` / `cpu` | 优先使用 GPU 加速训练 |
