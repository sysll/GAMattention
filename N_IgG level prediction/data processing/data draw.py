import pandas as pd
import numpy as np

# 读取原始Excel文件
input_file = 'D:\\Users\\ASUS\\Desktop\\论文\\N_lgG抗体水平检测\\数据集\\processed train_data without division.xlsx'
df = pd.read_excel(input_file)

# 随机选择800个样本的索引
random_indices = np.random.choice(len(df), size=800, replace=False)

# 获取随机样本
random_samples = df.iloc[random_indices]
# 将随机样本保存到另一个Excel文件
output_file = 'D:\\Users\\ASUS\\Desktop\\论文\\N_lgG抗体水平检测\\用于调参的模型\\N_used for train.xlsx'
random_samples.to_excel(output_file, index=False)






# 读取原始Excel文件
input_file = 'D:\\Users\\ASUS\\Desktop\\论文\\N_lgG抗体水平检测\\数据集\\processed test_data without division.xlsx'
df = pd.read_excel(input_file)

# 随机选择800个样本的索引
random_indices = np.random.choice(len(df), size=200, replace=False)

# 获取随机样本
random_samples = df.iloc[random_indices]
# 将随机样本保存到另一个Excel文件
output_file = 'D:\\Users\\ASUS\\Desktop\\论文\\N_lgG抗体水平检测\\用于调参的模型\\N_used for test.xlsx'
random_samples.to_excel(output_file, index=False)