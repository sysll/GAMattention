import numpy as np
from pygam import LinearGAM, s
import pandas as pd
# 创建源数据
X = np.random.random((1000, 9))  # 输入数据，大小为 (样本数, 特征数)
y = np.random.random((1000,))  # 目标标签，大小为 (样本数,)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
np.random.seed(8)

train_path = "N_used for train.xlsx"
test_path = "N_used for test.xlsx"


feature = [
    "年龄",
    "血_平均血红蛋白含量",
    "血_总胆固醇",
    "血_血小板计数",
    "血_平均RBC体积",
    "血_RBC分布宽度CV",
    "血_白细胞计数",
    "血_嗜酸细胞(#)",
    "血_红细胞计数"
]
df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)

X_train = df_train[feature]
Y_train = df_train['N_IgG']

X_test = df_test[feature]
Y_test = df_test['N_IgG']

X_train = np.array(X_train)
X_test = np.array(X_test)

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# 创建GAM模型并进行训练
gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8), lam=10, n_splines=10, tol=1e-5, max_iter=10, verbose=True).fit(X_train, Y_train)

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
