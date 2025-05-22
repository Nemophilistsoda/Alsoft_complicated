import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 数据加载
train = pd.read_csv('data/task5/train.csv')
test = pd.read_csv('data/task5/test.csv')

# 数据预处理
def preprocess(df):
    # 处理缺失值
    df = df.fillna(method='ffill')
    
    # 分类特征编码
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols)
    
    return df

# 应用预处理
train = preprocess(train)
test = preprocess(test)

# 准备训练数据
X = train.drop('label', axis=1)
y = train['label']

# 初始化XGBoost分类模型
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8
)

# 模型训练
model.fit(X, y)

# 测试集预测
test_pred = model.predict(test)

# 保存预测结果
with open('task5result.txt', 'w') as f:
    for pred in test_pred:
        f.write(f"{int(pred)}\n")