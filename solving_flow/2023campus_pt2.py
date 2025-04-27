from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

path = 'data/2023_campus_pt2.txt'
with open(path, 'r', encoding='utf-8') as f:
    documents = [line.strip() for line in f.readlines()]  # 移除lower()保持原始大小写

vec = TfidfVectorizer()
metrix = vec.fit_transform(documents)  # 计算TF-IDF矩阵
name = vec.get_feature_names_out()  # 获取所有特征词 并且按照字母顺序排序

# print(name)  # 输出特征词
# print(metrix)  # 输出TF-IDF矩阵

output_file ='result/2023_campus_pt2.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    # 输出的各行文本每个词的 TF-IDF 值 
    for i in range(len(documents)):
        feature_index = metrix[i].nonzero()[1]  # 非零元素的列索引
        tfidf_scores = zip(feature_index, [metrix[i, x] for x in feature_index])
        sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        doc_results = [(name[idx], round(score, 2)) for idx, score in sorted_scores]
        f.write(f"语句{i+1}：{doc_results}\n")

print('The result has been saved to 2023_campus_pt2.txt')

# 计算每行样本之间的余弦相似度（cosine_similarity）
cosine_similarity = np.dot(metrix, metrix.T)  # 计算余弦相似度矩阵
# print(cosine_similarity)  # 输出相似度矩阵
# 相似度矩阵是什么 有什么特点
'''
# 相似度矩阵是一个对称矩阵，对角线上的元素都是1，因为每个样本与自身的相似度是1。
# 相似度矩阵的非对角线上的元素是两个样本之间的相似度，取值范围是0到1。
# 相似度矩阵的元素值越大，表示两个样本越相似。
# 相似度矩阵的元素值越小，表示两个样本越不相似。
# 相似度矩阵的元素值为0，表示两个样本完全不相似。
# 相似度矩阵的元素值为1，表示两个样本完全相似。
'''
print(cosine_similarity.shape)
import matplotlib.pyplot as plt
plt.imshow(cosine_similarity, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()