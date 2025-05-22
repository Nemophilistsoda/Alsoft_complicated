import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 任务1：加载数据集
train = pd.read_csv('第三部分/train.csv')
test = pd.read_csv('第三部分/test.csv')

# 任务2：处理缺失值
for df in [train, test]:
    # 定量特征用中位数填充
    num_cols = ['Age', 'Fare']
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # 定性特征用众数填充
    cat_cols = ['Embarked']
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

# 任务3：删除无关特征
drop_cols = ['Cabin', 'Ticket', 'Name', 'PassengerId']
train = train.drop(drop_cols, axis=1)
test_passengerid = test['PassengerId']
test = test.drop(drop_cols, axis=1)

# 任务4：独热编码
encoder = OneHotEncoder()
cat_features = ['Pclass', 'Sex', 'Embarked']
encoded_train = encoder.fit_transform(train[cat_features]).toarray()
encoded_test = encoder.transform(test[cat_features]).toarray()

# 合并特征
train = pd.concat([train.drop(cat_features, axis=1), 
                 pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out())], axis=1)
test = pd.concat([test.drop(cat_features, axis=1),
                pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out())], axis=1)

# 任务5：划分训练集
X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# 任务6：训练随机森林
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# 任务7：绘制混淆矩阵
y_pred = rf.predict(X_val)
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 任务8：生成预测文件
test_pred = rf.predict(test)
result = pd.DataFrame({'PassengerId': test_passengerid, 'Survived': test_pred})
result.to_csv('user/Q3/result.csv', index=False)