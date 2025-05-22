import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 加载停用词
with open('data/task5/hit_stopwords.txt', 'r', encoding='gbk') as f:
    stopwords = set([line.strip() for line in f])

# 数据预处理函数
def preprocess(text):
    words = jieba.lcut(text)
    return ' '.join([w for w in words if w not in stopwords and len(w) > 1])

# 加载训练数据
train = pd.read_csv('data/task5/train_4_4000.txt', sep='\t', names=['text', 'label'])
train['processed'] = train['text'].apply(preprocess)

# 加载测试数据
test = pd.read_csv('data/task5/test_4_400.txt', sep='\t', names=['text'])
test['processed'] = test['text'].apply(preprocess)

# TF-IDF特征提取
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(train['processed'])
X_test = tfidf.transform(test['processed'])

y_train = train['label']

# 初始化随机森林模型
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)

# 评估训练集
train_pred = model.predict(X_train)
print(f'训练集准确率: {accuracy_score(y_train, train_pred):.4f}')

# 测试集预测
test_pred = model.predict(X_test)

# 保存预测结果
result = pd.DataFrame({
    'PassengerId': test.index,
    'Survived': test_pred
})
result.to_csv('result.csv', index=False)