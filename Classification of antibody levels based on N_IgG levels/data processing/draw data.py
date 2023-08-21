import pandas as pd
import numpy as np
import random
# 读取原始Excel文件
input_file = 'D:\\Users\\ASUS\\Desktop\\论文\\基于S1_lgG的阴阳分类\\数据处理\\processed train_data without division.xlsx'
df = pd.read_excel(input_file)
# 设置条件，筛选满足条件的样本数量
desired_count = 400+random.randint(-10, 10)

# 获取满足条件的索引
condition_indices = df[df['S1_IgG'] == 1].index

# 获取满足条件的样本索引，并随机选择指定数量的索引
random_indices = np.random.choice(condition_indices, size=desired_count, replace=False)

# 获取随机样本(400+random.randint(-10, 10)个标签是1的
random_samples1 = df.loc[random_indices]

""""""
desired_count = 800-desired_count
condition_indices = df[df['S1_IgG'] == 0].index

# 获取满足条件的样本索引，并随机选择指定数量的索引
random_indices = np.random.choice(condition_indices, size=desired_count, replace=False)

# 获取随机样本(400+random.randint(-10, 10)个标签是1的
random_samples2 = df.loc[random_indices]

merged_df = pd.concat([random_samples1, random_samples2])
#打乱
shuffled_df = merged_df.sample(frac=1, random_state=42)
shuffled_df = shuffled_df.reset_index(drop=True)
# 将随机样本保存到另一个Excel文件
output_file = 'D:\\Users\\ASUS\\Desktop\\论文\\基于S1_lgG的阴阳分类\\用于调参的模型\\S1_used for train.xlsx'
shuffled_df.to_excel(output_file, index=False)






# 读取原始Excel文件
input_file = 'D:\\Users\\ASUS\\Desktop\\论文\\基于S1_lgG的阴阳分类\\数据处理\\processed test_data without division.xlsx'
df = pd.read_excel(input_file)

# 随机选择800个样本的索引
random_indices = np.random.choice(len(df), size=200, replace=False)

# 获取随机样本
random_samples = df.iloc[random_indices]
# 将随机样本保存到另一个Excel文件
output_file = 'D:\\Users\\ASUS\\Desktop\\论文\\基于S1_lgG的阴阳分类\\用于调参的模型\\S1_used for test.xlsx'
random_samples.to_excel(output_file, index=False)