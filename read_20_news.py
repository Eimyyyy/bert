from sklearn.datasets import fetch_20newsgroups
import numpy as np

categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_subset = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

unique, counts = np.unique(newsgroups_subset.target, return_counts=True)
for idx, cat in enumerate(newsgroups_subset.target_names):
    print(f"类别 '{cat}' 的样本数: {counts[idx]}")

first_atheism_idx = list(newsgroups_subset.target).index(0)
first_christian_idx = list(newsgroups_subset.target).index(0)

print("\n alt.atheism的内容")
print(newsgroups_subset.data[first_atheism_idx][:1000])
print("\n soc.religion.christian 的内容")
print(newsgroups_subset.data[first_christian_idx])