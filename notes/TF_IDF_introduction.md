# TF-IDF Introduction
documents = [
    "this is the first document",
    "this document is the second document",
    "and this is the third one",
    "is this the first document"
]
先用这行代码 把工具构建给vec vec = TfidfVectorize() 
然后用vec.fit_transform(documents) 把documents 变成一个矩阵 这个矩阵的每一行代表一个document 每一列代表一个word 这个矩阵的元素代表这个word在这个document中出现的次数
然后用vec.get_feature_names() 可以得到这个矩阵的列名 也就是这个矩阵的每一列代表的word
对于documents 这里的feature_names 是['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this'] 注意到 这里的单词是按照字母顺序排列的 
打印这个矩阵 可以得到
(0, 1)    0.4698  # 第1篇文档中"document"的TF-IDF值
(0, 2)    0.5803  # 第1篇文档中"first"的TF-IDF值
(1, 1)    0.6876  # 第2篇文档中"document"的TF-IDF值
*** 其中元组 (i,j) 表示第i篇文档的第j个特征词 ***

