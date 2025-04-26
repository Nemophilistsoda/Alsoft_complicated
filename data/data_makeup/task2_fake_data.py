import numpy as np
import pandas as pd

# 生成1000行伪造数据
np.random.seed(42)  # 固定随机种子保证可复现
mu1, sigma1 = 50, 10  # sensor_1的均值和标准差
mu2, sigma2 = 30, 5   # sensor_2的均值和标准差

data = {
    "timestamp": pd.date_range(start="2024-01-01", periods=1000, freq="T"),
    "sensor_1": np.random.normal(mu1, sigma1, 1000),
    "sensor_2": np.random.normal(mu2, sigma2, 1000)
}

df = pd.DataFrame(data)
df.to_csv("data/task2_fake_datanew.csv", index=False)
print("伪造数据文件已生成：task2_fake_datanew.csv")