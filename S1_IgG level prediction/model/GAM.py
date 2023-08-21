import numpy as np
from pygam import LinearGAM
import pandas as pd
# 创建源数据
X = np.random.random((1000, 9))  # 输入数据，大小为 (样本数, 特征数)
y = np.random.random((1000,))  # 目标标签，大小为 (样本数,)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
np.random.seed(8)

train_path = "used for train.xlsx"
test_path = "used for test.xlsx"
#随机森林特征
feature = ['年龄', '血_血小板压积', '血_RBC分布宽度SD', '血_RBC分布宽度CV', '血_碳酸氢根', '血_eGFR(基于CKD-EPI方程)', '血_红细胞压积', '血_球蛋白', '血_红细胞计数']

df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)

X_train = df_train[feature]
Y_train = df_train['S1_IgG']

X_test = df_test[feature]
Y_test = df_test['S1_IgG']

X_train = np.array(X_train)
X_test = np.array(X_test)

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# 创建GAM模型并进行训练
gam = LinearGAM().fit(X_train, Y_train)

# 预测评价集数据
y_pred = gam.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)

# 计算平均绝对误差
mae = mean_absolute_error(Y_test, y_pred)

# 计算均方根误差
rmse = mean_squared_error(Y_test, y_pred, squared=False)

# 计算R
correlation_matrix = np.corrcoef(Y_test, y_pred)
r = correlation_matrix[0, 1]

# 打印结果
print("MSE", mse)
print("MAE", mae)
print("RMSE", rmse)
print("R^2", r)
