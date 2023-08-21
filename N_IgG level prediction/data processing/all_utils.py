import numpy as np
import pandas as pd
def division_X(df):
    # dataFrame和list之间相互转换
    data = df.values.tolist()

    list_new = []
    list_short = []
    # [0,len(data)-1) == [0,len(data)-2]
    for i in range(0, len(data) - 1):
        # 如果i和i+1的ID相同,那就将该条样本添加到list_short中
        if data[i][0] == data[i + 1][0]:
            list_short.append(data[i])
        # 否则将list_short添加到list_new中,并且重置list_short(便于存下一个ID病人的信息)
        else:
            list_short.append(data[i])
            list_new.append(list_short)
            list_short = []

        if i == len(data) - 2:
            list_short.append(data[i + 1])
            list_new.append(list_short)

    return list_new



def delet_and_replace(list_new):
        # 删除只有一个ID的人
        list_new = [group for group in list_new if len(group) > 1]

        # 替换缺失值为相同 ID 的其他值的平均数
        for group in list_new:
            for n in range(len(group[0]) - 1):
                values = [x[n + 1] for x in group]
                mean_value = np.nanmean(values)  # 使用 np.nanmean() 计算平均值，忽略 NaN 值
                for i in range(len(group)):
                    if np.isnan(group[i][n + 1]):
                        group[i][n + 1] = mean_value
        return list_new