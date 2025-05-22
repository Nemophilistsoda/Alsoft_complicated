import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# 数据加载
train = pd.read_csv('data/task5/train.csv')
val = pd.read_csv('data/task5/val.csv')
test = pd.read_csv('data/task5/test.csv')

# 数据预处理
def preprocess(df):
    # 处理缺失值
    df = df.fillna({'电梯情况': '无', '装修情况': '其他'})
    
    # 特征工程
    df['建筑年份'] = 2023 - df['建筑时间']
    
    # 类别特征编码
    cat_cols = ['电梯情况', '区域', '装修情况']
    df = pd.get_dummies(df, columns=cat_cols)
    
    return df

# 应用预处理
train = preprocess(train)
val = preprocess(val)
test = preprocess(test)

# 准备训练数据
X_train = train.drop('价格', axis=1)
y_train = train['价格']

# 初始化XGBoost回归模型
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    early_stopping_rounds=50
)

# 模型训练
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (val.drop('价格', axis=1), val['价格'])],
          verbose=50)

# 测试集预测
test_pred = model.predict(test)

# 保存预测结果
with open('task5result.txt', 'w') as f:
    for pred in test_pred:
        f.write(f"{int(pred)}\n")