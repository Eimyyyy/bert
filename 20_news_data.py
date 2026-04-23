import re
import string
import pickle
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  # 新增：用于划分数据集


def preprocess_text(text):
    """简单的文本预处理"""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text


def build_vocab(texts):
    """构建词汇表"""
    word_freq = Counter()
    for text in texts:
        words = text.split()
        word_freq.update(words)

    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.items():
        if freq >= 2:
            word_to_idx[word] = len(word_to_idx)
    return word_to_idx


def load_and_preprocess_data():
    """加载并预处理20newsgroups数据，并划分出验证集"""
    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,
                                          remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

    X_train_full = [preprocess_text(doc) for doc in newsgroups_train.data]
    X_test = [preprocess_text(doc) for doc in newsgroups_test.data]

    label_encoder = LabelEncoder()
    y_train_full = label_encoder.fit_transform(newsgroups_train.target)
    y_test = label_encoder.transform(newsgroups_test.target)

    # 将训练集划分为 训练集(80%) 和 验证集(20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )


    word_to_idx = build_vocab(X_train + X_val + X_test)
    vocab_size = len(word_to_idx)

    print(f"词汇表大小: {vocab_size}")
    print(f"训练集样本数量: {len(y_train)}")
    print(f"验证集样本数量: {len(y_val)}")
    print(f"测试集样本数量: {len(y_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, word_to_idx, vocab_size


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, word_to_idx, vocab_size = load_and_preprocess_data()

    with open('20news_processed.pkl', 'wb') as f:
        pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test, word_to_idx, vocab_size), f)

    print("数据已成功保存到 20news_processed.pkl")