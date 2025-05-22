import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

# 数据加载
train = pd.read_csv('data/task5/train.txt', sep='\t', names=['text', 'label'])
test = pd.read_csv('data/task5/test.txt', names=['text'])

# 文本预处理
def preprocess(text):
    text = text.replace('\n', '').replace('\r', '')
    words = jieba.lcut(text)
    return ' '.join(words)

train['text'] = train['text'].apply(preprocess)
test['text'] = test['text'].apply(preprocess)

# TF-IDF特征提取
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(train['text'])
X_test = tfidf.transform(test['text'])

# 模型训练
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, train['label'])

# 预测结果
test_pred = model.predict(X_test)

# 结果输出
with open('task5result.txt', 'w') as f:
    for pred in test_pred:
        f.write(f"{pred}\n")