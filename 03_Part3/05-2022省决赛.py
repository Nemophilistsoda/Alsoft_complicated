import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 任务1：加载数据集
data = pd.read_csv('第三部分/task3data.csv', header=None)
X = data.iloc[:, :30]
y = data.iloc[:, 30]

# 任务2：Z-score标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 任务3：数据集切分
X_train, X_test = X_scaled[:500], X_scaled[500:]
y_train, y_test = y[:500], y[500:]

# 任务4：训练决策树模型
dtc = DecisionTreeClassifier(max_depth=10)
dtc.fit(X_train, y_train)

# 任务5：测试集评估
y_pred = dtc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'测试集分类准确率: {accuracy:.4f}')