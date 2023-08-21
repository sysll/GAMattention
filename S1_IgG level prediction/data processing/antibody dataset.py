
from all_utils import division_X,delet_and_replace
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import numpy as np
def del_columns(data):
    t = int(0.7 * data.shape[0])
    data = data.dropna(thresh=t, axis=1)  # 保留至少有70%个非空的列
    return data
def del_rows(data):
    t = int(0.5 * data.shape[1])
    data = data.dropna(thresh=t)  # 保留至少有50%非空的行
    return data
def list_to_dataframe(list_new):
    # 将list_new转换为DataFrame
    df = pd.DataFrame([item for sublist in list_new for item in sublist])

    return df

# 设定随机种子为5
seed = 5
df = pd.read_excel(r'D:\\Users\\ASUS\\Desktop\\原始数据.xlsx')

#丢掉一些无用
df = df.drop(['首发症状','临床症状','样本ID','核酸-汇总','呼吸频率(RR)','动脉血氧分压（PaO2)','氧饱和度%（SpO2)','吸氧流量（面罩或鼻导管）L/min','FiO2(%)'],axis = 1)

#临床结局  严重程度（最终） N_IgG  是否进入ICU  发病天数 这几个特征和水平相关性强，所以去除掉
df = df.drop(['发病日期', '入院时间', '出院/死亡时间', '检测日期', '临床结局 ', '严重程度（最终）', 'N_IgG', '是否进入ICU', '发病天数'], axis=1)

# 去掉存在部分'S1_IgG'缺失的行
df = df.dropna(axis=0,subset=['S1_IgG'], inplace=False)

# 字符型变成数值型
df['性别'] = df['性别'].astype(str).map({'女': 0, '男': 1})

# 输出缺失值
print('数据大小:', df.shape)
df = del_columns(df)
print('列处理后,数据大小:', df.shape)
df = del_rows(df)
print('行处理后,数据大小:', df.shape)



#因为这两列数据里面有int，有float
df['血_RBC分布宽度CV'] = pd.to_numeric(df['血_RBC分布宽度CV'], errors='coerce')
df['血_RBC分布宽度SD'] = pd.to_numeric(df['血_RBC分布宽度SD'], errors='coerce')
df['血_嗜碱细胞(%)'] = pd.to_numeric(df['血_嗜碱细胞(%)'], errors='coerce')
df['血_嗜碱细胞(#)'] = pd.to_numeric(df['血_嗜碱细胞(#)'], errors='coerce')
df['血_血小板计数'] = pd.to_numeric(df['血_血小板计数'], errors='coerce')
df['血_中性粒细胞(%)'] = pd.to_numeric(df['血_中性粒细胞(%)'], errors='coerce')
df['血_中性粒细胞(#)'] = pd.to_numeric(df['血_中性粒细胞(#)'], errors='coerce')
df['血_单核细胞(%)'] = pd.to_numeric(df['血_单核细胞(%)'], errors='coerce')
df['血_淋巴细胞(%)'] = pd.to_numeric(df['血_淋巴细胞(%)'], errors='coerce')
df['血_淋巴细胞(#)'] = pd.to_numeric(df['血_淋巴细胞(#)'], errors='coerce')



df2 = division_X(df)
#对缺失值按相同ID补平均数
df2 = delet_and_replace(df2)


# #因为split是随机抽取，抽的是乱的。所以要从新排一些顺序
train_temp, test_temp = train_test_split(df2, test_size=0.2, random_state=seed)
train_temp = sorted(train_temp, key=lambda x: x[0][0], reverse=False)
test_temp = sorted(test_temp, key=lambda x: x[0][0], reverse=False)


train2 = []
for i in train_temp:
    for j in i:
        train2.append(j)


train = pd.DataFrame(train2, columns=df.columns)
#在这丢掉病人的ID是为了初始化
train_drop = train.drop(['病人ID', '性别', '高血压(0=无，1=有)', '糖尿病(0=无，1=有)'], axis=1)

sc1 = StandardScaler()
features = sc1.fit_transform(train_drop)
data_transform = pd.DataFrame(features, columns=train_drop.columns)

#对都没有的用所有的平均数去补
data_transform = data_transform.fillna(data_transform.median(numeric_only=True))

data_transform.insert(0, '糖尿病(0=无，1=有)', train['糖尿病(0=无，1=有)'].tolist())
data_transform.insert(0, '高血压(0=无，1=有)', train['高血压(0=无，1=有)'].tolist())
data_transform.insert(0, '性别', train['性别'].tolist())
data_transform.insert(0, '病人ID', train['病人ID'].tolist())
data_transform.to_excel(
    "D:\\Users\\ASUS\\Desktop\\论文\\抗体水平检预测\\数据处理\\processed train_data without division.xlsx",
    engine="openpyxl")

test2 = []
for i in test_temp:
    for j in i:
        test2.append(j)

test = pd.DataFrame(test2, columns=df.columns)

test_data = test.drop(['病人ID', '性别', '高血压(0=无，1=有)', '糖尿病(0=无，1=有)'], axis=1)

features = sc1.transform(test_data)
data_transform = pd.DataFrame(features, columns=test_data.columns)
#对都没有的用所有的平均数去补
data_transform = data_transform.fillna(data_transform.median(numeric_only=True))

data_transform.insert(0, '糖尿病(0=无，1=有)', test['糖尿病(0=无，1=有)'].tolist())
data_transform.insert(0, '高血压(0=无，1=有)', test['高血压(0=无，1=有)'].tolist())
data_transform.insert(0, '性别', test['性别'].tolist())
data_transform.insert(0, '病人ID', test['病人ID'].tolist())
data_transform.to_excel(
    "D:\\Users\\ASUS\\Desktop\\论文\\抗体水平检预测\\数据处理\\processed test_data without division.xlsx",
    engine="openpyxl")


