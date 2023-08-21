
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor


np.random.seed(4)

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

y_train = Y_train
y_test = Y_test
# param_grid = {
#     'n_estimators': 600,
#     'max_features': 1,
#     'max_depth': 10,
#     'min_samples_split': 8,
#     'min_samples_leaf': 5
# }
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







