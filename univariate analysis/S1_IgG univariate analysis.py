import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pandas
from scipy.stats import linregress

#九个特征对应的图
df1 = pd.read_excel("D:\\Users\\ASUS\\Desktop\\论文\\抗体水平检预测\\数据处理\\processed train_data without division.xlsx")
df2 = pd.read_excel("D:\\Users\\ASUS\\Desktop\\论文\\抗体水平检预测\\数据处理\\processed test_data without division.xlsx")
df = pd.concat([df1, df2], ignore_index=True)
S1_IgG = list(df['S1_IgG'])
age = list(df["年龄"])
PCT = list(df['血_血小板压积'])
RDW_SD = list(df['血_RBC分布宽度SD'])
RDW_CV = list(df['血_RBC分布宽度CV'])
HCO3 = list(df['血_碳酸氢根'])
eGFR = list(df['血_eGFR(基于CKD-EPI方程)'])
HCT = list(df['血_红细胞压积'])
GlOB = list(df['血_球蛋白'])
RBC = list(df['血_红细胞计数'])

name = ['(1) Age', '(2) PCT', '(3) RDW_SD', '(4) RDW_CV', '(5) HCO3-', '(6) eGFR(CKD-EP)', '(7) HCT', '(8) GlOB', '(9) RBC']
namelist = [age, PCT, RDW_SD, RDW_CV , HCO3, eGFR, HCT, GlOB, RBC]

def all_paint(name, i):
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    train_data = pd.DataFrame({name: namelist[i], 'S1_IgG': S1_IgG})
    plt.subplot(3, 3, i +1)
    # 画带有置信区间的回归图
    if i == 0 or i == 1 or i == 3 or i == 7 :
        sns.regplot(x=name, y='S1_IgG', data=train_data, scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3}, fit_reg=False)
        # 进行线性回归拟合
        scatter_color = sns.color_palette()[0]
        coefficients = np.polyfit(train_data[name], train_data['S1_IgG'], 1)

        # 生成拟合线的数据点
        x_fit = np.linspace(-8, 8, 100)
        y_fit = np.polyval(coefficients, x_fit)

        # 添加拟合线到图形中
        plt.plot(x_fit, y_fit, color=scatter_color, linestyle='-', linewidth=1.5)

    else:
        sns.regplot(x=name, y='S1_IgG', data=train_data, scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3}, fit_reg=False)
    plt.xlim(-6, 6)
    plt.ylim(-2, 4)
#画全部数据的特征相关图
for i in range(9):
    all_paint(name[i], i)
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


# #以下是做第一天的数据的数据处理
# df = Get_first_day(df)
# S1_IgG = list(df['S1_IgG'])
# age = list(df["年龄"])
# MPV = list(df['血_血小板压积'])
# RDW_SD = list(df['血_RBC分布宽度SD'])
# RDW_CV = list(df['血_RBC分布宽度CV'])
# HCO3 = list(df['血_碳酸氢根'])
# eGFR = list(df['血_eGFR(基于CKD-EPI方程)'])
# HCT = list(df['血_红细胞压积'])
# GlOB = list(df['血_球蛋白'])
# RBC = list(df['血_红细胞计数'])
#
# name = ['Age', 'MPV', 'RDW_SD', 'RDW_CV', 'HCO3', 'eGFR(CKD-EP)', 'HCT', 'GlOB', 'RBC']
# namelist = [age, MPV, RDW_SD, RDW_CV , HCO3, eGFR, HCT, GlOB, RBC]
#
def paint_part(name, i):
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    train_data = pd.DataFrame({name: namelist[i], 'S1_IgG': S1_IgG})
    # 计算 PCC
    pcc = train_data[name].corr(train_data['S1_IgG'])
    plt.subplot(3, 3, i + 1)

    # 画带有置信区间的回归图
    sns.regplot(x=name, y='S1_IgG', data=train_data, scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
                line_kws={'color': 'k'})

    # 回归拟合
    slope, intercept, _, _, _ = linregress(namelist[i], S1_IgG)

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





# # #以下是对出院的数据
# df = Get_Last_day(df)
# S1_IgG = list(df['S1_IgG'])
# age = list(df["年龄"])
# MPV = list(df['血_血小板压积'])
# RDW_SD = list(df['血_RBC分布宽度SD'])
# RDW_CV = list(df['血_RBC分布宽度CV'])
# HCO3 = list(df['血_碳酸氢根'])
# eGFR = list(df['血_eGFR(基于CKD-EPI方程)'])
# HCT = list(df['血_红细胞压积'])
# GlOB = list(df['血_球蛋白'])
# RBC = list(df['血_红细胞计数'])
#
# name = ['Age', 'MPV', 'RDW_SD', 'RDW_CV', 'HCO3', 'eGFR(CKD-EP)', 'HCT', 'GlOB', 'RBC']
# namelist = [age, MPV, RDW_SD, RDW_CV , HCO3, eGFR, HCT, GlOB, RBC]


# for i in range(9):
#     paint_part(name[i], i)
# plt.show()




