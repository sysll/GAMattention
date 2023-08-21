
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor




train_path = "used for train.xlsx"
test_path = "used for test.xlsx"
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

y_train = Y_train
y_test = Y_test

# 创建随机森林回归模型
rf_model = RandomForestRegressor()

# 拟合模型
rf_model.fit(X_train, y_train)
# 预测
y_pred = rf_model.predict(X_test)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
correlation_matrix = np.corrcoef(y_test, y_pred)
r = correlation_matrix[0, 1]

# 打印评估指标
print("MSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)
print("R:", r)

import matplotlib.pyplot as plt
plt.plot(range(len(y_test)), y_test, color='b', label='Actual')
plt.plot(range(len(y_test)), y_pred, color='r', label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.title('Random Forest Regression - Actual vs. Predicted')
plt.legend()
plt.show()







