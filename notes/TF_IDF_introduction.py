import math
from sklearn.feature_extraction.text import TfidfVectorizer

# 示例文档集合
documents = [
    "this is the first document",
    "this document is the second document",
    "and this is the third one",
    "is this the first document"
]

vec = TfidfVectorizer()
matrix = vec.fit_transform(documents)  # 计算TF-IDF矩阵
feature_names = vec.get_feature_names_out()  # 获取所有特征词 并且按照字母顺序排序
print(feature_names)  # 输出特征词
print(matrix)
