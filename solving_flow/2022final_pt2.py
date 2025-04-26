import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

file_path = 'data/2022final_pt2/2022final_pt2_fakedata_task1.txt'
with open(file_path, 'r', encoding='gbk') as f:
    documents = [line.strip() for line in f.readlines()]  # 移除lower()保持原始大小写

# 配置TF-IDF计算器（使用默认参数）
vec = TfidfVectorizer(tokenizer=lambda x: x.split(),  # 已分好词，按空格分割
                     lowercase=False)  # 不自动转小写

# 计算TF-IDF矩阵（稀疏矩阵格式）
tfidf_matrix = vec.fit_transform(documents)  
feature_names = vec.get_feature_names_out()  # 获取所有特征词

# 整理并排序结果
results = []
for i in range(len(documents)):
    # 获取当前文档的所有特征词和对应TF-IDF值
    feature_index = tfidf_matrix[i].nonzero()[1]  # 非零元素的列索引
    tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
    
    # 按TF-IDF值降序排序
    sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    
    # 格式化为要求的输出形式
    doc_results = [(feature_names[idx], round(score, 2)) for idx, score in sorted_scores]
    results.append(doc_results)
    print(f"语句{i+1}：{doc_results}")

# 保存结果到文件
output_file = 'result/2022final_pt2.txt'
with open(output_file, 'w', encoding='gbk') as f:
    for i, doc_results in enumerate(results):
        f.write(f"语句{i+1}：{doc_results}\n")
print('The result has been saved to 2022final_pt2.txt')