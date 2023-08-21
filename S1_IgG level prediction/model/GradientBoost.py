from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
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

best_params = {'learning_rate': 0.01, 'max_depth': 5, 'max_features': 1.0, 'min_samples_leaf': 2,
               'min_samples_split': 2, 'n_estimators': 300, 'subsample': 0.8}

# 创建GradientBoostingRegressor对象
regressor = GradientBoostingRegressor(**best_params)

# 在训练集上拟合模型
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)

# 计算平均绝对误差
mae = mean_absolute_error(Y_test, Y_pred)

# 计算均方根误差
rmse = mean_squared_error(Y_test, Y_pred, squared=False)

# 计算R
correlation_matrix = np.corrcoef(Y_test, Y_pred)
r = correlation_matrix[0, 1]

# 打印结果
print("MSE", mse)
print("MAE", mae)
print("RMSE", rmse)
print("R^2", r)

from matplotlib import pyplot as plt
x = range(len(Y_test))

# 绘制真实值和预测值的折线图
plt.plot(x, Y_test, label='True')
plt.plot(x, Y_pred, label='pred')

# 添加图例和标签
plt.legend()
plt.xlabel('样本')
plt.ylabel('数值')

# 显示图形
plt.show()