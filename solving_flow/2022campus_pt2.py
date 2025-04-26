import pandas as pd

df = pd.read_csv('data/task2_fake_datanew.csv')
# print(df.head())

# 分别求取sensor_1，sensor_2两个特征的均值和方差
# res_mean = df[['sensor_1', 'sensor_2']].mean()
# res_var = df[['sensor_1', 'sensor_2']].var()
# print(res_mean)
# print(res_var)
sensor_1_mean = df['sensor_1'].mean()
sensor_2_mean = df['sensor_2'].mean()
sensor_1_var = df['sensor_1'].var()
sensor_2_var = df['sensor_2'].var()

# 对sensor_1，sensor_2 特征进行Z-score标准化，
data = df[['sensor_1', 'sensor_2']]
data_zscore = (data - data.mean()) / data.std()
print(data_zscore.head())
print(data_zscore.mean())
# 输出标准化之后的sensor_1，sensor_2 的均值和方差。
sensor_1_mean_new = data_zscore['sensor_1'].mean()
sensor_2_mean_new = data_zscore['sensor_2'].mean()
sensor_1_var_new = data_zscore['sensor_1'].var()
sensor_2_var_new = data_zscore['sensor_2'].var()

output_file = 'result/2022campus_pt2.csv'
with open(output_file, 'w') as f:
    # f.write('sensor_1_mean,sensor_1_var,sensor_2_mean,sensor_2_var\n')
    f.write(f'{sensor_1_mean},{sensor_1_var},{sensor_2_mean},{sensor_2_var}\n')
    f.write(f'{sensor_1_mean_new},{sensor_1_var_new},{sensor_2_mean_new},{sensor_2_var_new}\n')
print('The result has been saved to 2022campus_pt2.csv')