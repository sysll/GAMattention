import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pandas
from scipy.stats import linregress
#九个特征对应的图
df1 = pd.read_excel("D:\\Users\\ASUS\\Desktop\\论文\\N_lgG抗体水平检测\\数据集\\processed train_data without division.xlsx")
df2 = pd.read_excel("D:\\Users\\ASUS\\Desktop\\论文\\N_lgG抗体水平检测\\数据集\\processed test_data without division.xlsx")
df = pd.concat([df1, df2], ignore_index=True)
N_IgG = list(df['N_IgG'])
Age = list(df["年龄"])
HB= list(df["血_平均血红蛋白含量"])
TC= list(df["血_总胆固醇"])
PLT= list(df["血_血小板计数"])
MCV= list(df["血_平均RBC体积"])
RDW_CV= list(df["血_RBC分布宽度CV"])
WBC= list(df["血_白细胞计数"])
Eos= list(df["血_嗜酸细胞(#)"])
RBC= list(df["血_红细胞计数"])
name = ["(1) Age", "(2) MCH", "(3) TC", "(4) PLT", "(5) MCV", "(6) RDW_CV", "(7) WBC", "(8) Eos(#)", "(9) RBC"]
namelist = [Age,HB,TC,PLT,MCV,RDW_CV,WBC,Eos,RBC]


def paint(name, i):
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    train_data = pd.DataFrame({name: namelist[i], 'N_IgG': N_IgG})
    plt.subplot(3, 3, i +1)
    # 画带有置信区间的回归图
    if i == 0 or i == 3 or i == 5 or i == 8:
        sns.regplot(x=name, y='N_IgG', data=train_data, scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3}, fit_reg=False)
        # 进行线性回归拟合
        coefficients = np.polyfit(train_data[name], train_data['N_IgG'], 1)

        # 生成拟合线的数据点
        x_fit = np.linspace(-8, 8, 100)
        y_fit = np.polyval(coefficients, x_fit)

        # 添加拟合线到图形中
        plt.plot(x_fit, y_fit, color=(0.1, 0.6, 0.8), linestyle='-', linewidth=2)

    else:
        sns.regplot(x=name, y='N_IgG', data=train_data, scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3}, fit_reg=False)
    plt.xlim(-6, 6)
    plt.ylim(-2, 4)
# 画全部数据的特征相关图
for i in range(9):
    paint(name[i], i)
plt.show()











# #一致性检验图
def Get_ID_length_list(ID_list:list):
    buffer_ID = ''
    temp_number = 0
    length_list = []
    for ID in ID_list:
        if buffer_ID != ID:
            length_list.append(temp_number)
            temp_number = 1
            buffer_ID = ID
        else:
            temp_number +=1
    length_list.append(temp_number)
    we_want = length_list[1:]
    return we_want

#Length_list:list,
def Get_first_day(df:pandas.DataFrame):  #得到第一天入院的病人情况
    Length_list = Get_ID_length_list(ID_list=list(df['病人ID']))
    drop_index = -1
    max_i = 0
    for i in Length_list:
            # 用来判断是不是只有一个，或者判断是不是第一个
            max_i = i
            while i > 0:
                # 向后滚动
                drop_index += 1
                # 如果是第一个之后的，或者是只有一个的，就去掉。
                if i < max_i or i == 1:
                    df = df.drop(labels=drop_index,axis=0)
                i -= 1
    return df

def Get_Last_day( df:pandas.DataFrame):#得到最后一天入院的病人的情况
    Length_list = Get_ID_length_list(ID_list=list(df['病人ID']))
    drop_index = -1
    max_i = 0
    for i in Length_list:
            # 用来判断是不是只有一个，或者判断是不是第一个
        max_i = i
        while i > 0:
            # 向后滚动
            drop_index += 1
            # 只变了这里
            if i >1 or max_i == 1:
                df = df.drop(labels=drop_index,axis=0)
            i -= 1
    return df

def Get_Middle_day(Length_list:list, df:pandas.DataFrame):#得到中间数据
    Length_list = Get_ID_length_list(ID_list=list(df['病人ID']))
    drop_index = -1
    max_i = 0
    for i in Length_list:
            # 用来判断是不是只有一个，或者判断是不是第一个
            max_i = i
            while i > 0:
                # 向后滚动
                drop_index += 1
                # 如果是第一个之后的，或者是只有一个的，就去掉。
                if i==1 or max_i == 1 or i == max_i:
                    df = df.drop(labels=drop_index,axis=0)
                i -= 1
    return df


#以下是做第一天的数据的数据处理
# df = Get_first_day(df)
# N_IgG = list(df['N_IgG'])
# Age = list(df["年龄"])
# HB= list(df["血_平均血红蛋白含量"])
# TC= list(df["血_总胆固醇"])
# PLT= list(df["血_血小板计数"])
# MCV= list(df["血_平均RBC体积"])
# RDW_CV= list(df["血_RBC分布宽度CV"])
# WBC= list(df["血_白细胞计数"])
# Eos= list(df["血_嗜酸细胞(#)"])
# RBC= list(df["血_红细胞计数"])
# name = ["Age", "HB", "TC", "PLT", "MCV", "RDW_CV", "WBC", "Eos(#)", "RBC"]
# namelist = [Age,HB,TC,PLT,MCV,RDW_CV,WBC,Eos,RBC]

# def paint_part(name, i):
#     plt.subplots_adjust(hspace=0.3, wspace=0.3)
#     train_data = pd.DataFrame({name: namelist[i], 'N_IgG': N_IgG})
#     # 计算 PCC
#     pcc = train_data[name].corr(train_data['N_IgG'])
#     plt.subplot(3, 3, i + 1)
#
#     # 画带有置信区间的回归图
#     sns.regplot(x=name, y='N_IgG', data=train_data, scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
#                 line_kws={'color': 'k'})
#
#     # 回归拟合
#     slope, intercept, _, _, _ = linregress(namelist[i], N_IgG)
#
#     # 计算置信区间
#     n = len(namelist[i])  # 样本数量
#     t = 2.262  # 95%置信水平下自由度为 n-2 的 t 分布的临界值
#     x_mean = np.mean(namelist[i])
#     x_std = np.std(namelist[i], ddof=1)
#     t_value = t * x_std / np.sqrt(n)
#     slope_lower = slope - t_value
#     slope_upper = slope + t_value
#
#     # 在图片上面标上面标上pcc和W_bound
#     plt.text(0.05, 0.90, f'PCC: {pcc:.2f}', transform=plt.gca().transAxes)
#     plt.text(0.05, 0.80, f'W_bound({slope_lower:.2f}', transform=plt.gca().transAxes)
#     plt.text(0.37, 0.80, f', {slope_upper:.2f})', transform=plt.gca().transAxes)
# for i in range(9):
#     paint_part(name[i], i)
# plt.show()











# # #以下是对出院的数据
# df = Get_Last_day(df)
# N_IgG = list(df['N_IgG'])
# Age = list(df["年龄"])
# MCH= list(df["血_平均血红蛋白含量"])
# TC= list(df["血_总胆固醇"])
# PLT= list(df["血_血小板计数"])
# MCV= list(df["血_平均RBC体积"])
# RDW_CV= list(df["血_RBC分布宽度CV"])
# WBC= list(df["血_白细胞计数"])
# Eos= list(df["血_嗜酸细胞(#)"])
# RBC= list(df["血_红细胞计数"])
#
# name = ["Age", "MCH", "TC", "PLT", "MCV", "RDW_CV", "WBC", "Eos(#)", "RBC"]
# namelist = [Age,MCH,TC,PLT,MCV,RDW_CV,WBC,Eos,RBC]
def paint_part(name, i):
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    train_data = pd.DataFrame({name: namelist[i], 'N_IgG': N_IgG})
    # 计算 PCC
    pcc = train_data[name].corr(train_data['N_IgG'])
    plt.subplot(3, 3, i + 1)

    # 画带有置信区间的回归图
    sns.regplot(x=name, y='N_IgG', data=train_data, scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
                line_kws={'color': 'k'})

    # 回归拟合
    slope, intercept, _, _, _ = linregress(namelist[i], N_IgG)

    # 计算置信区间
    n = len(namelist[i])  # 样本数量
    t = 2.262  # 95%置信水平下自由度为 n-2 的 t 分布的临界值
    x_mean = np.mean(namelist[i])
    x_std = np.std(namelist[i], ddof=1)
    t_value = t * x_std / np.sqrt(n)
    slope_lower = slope - t_value
    slope_upper = slope + t_value

    # 在图片上面标上面标上pcc和W_bound
    plt.text(0.05, 0.90, f'PCC: {pcc:.2f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.80, f'W_bound({slope_lower:.2f}', transform=plt.gca().transAxes)
    plt.text(0.37, 0.80, f', {slope_upper:.2f})', transform=plt.gca().transAxes)

# for i in range(9):
#     paint_part(name[i], i)
# plt.show()