import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
#scad回归
# 生成示例数据
torch.manual_seed(33)
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

X_train = torch.DoubleTensor(X_train)
X_test = torch.DoubleTensor(X_test)
Y_train = torch.DoubleTensor(Y_train)
y_test = torch.DoubleTensor(Y_test)


# 定义回归模型
class RegressionModel(nn.Module):
    def __init__(self, num_features):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(num_features, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


# 定义SCAD惩罚函数
def scad( beta):
    lambda_ = 1
    a = 3.7
    alpha = 1
    penalty = 0
    if beta == 0:
        penalty += 0
    elif torch.abs(beta) <= lambda_:
        penalty += alpha * lambda_ * torch.abs(beta)
    elif torch.abs(beta) <= a * lambda_:
        penalty += (lambda_ * (a * alpha - 1) * torch.abs(beta) -
                    (lambda_ ** 2) * (alpha - 1) / 2) / (a - 1)
    else:
        penalty += lambda_ ** 2 * (a + 1) * alpha / 2
    return penalty



# 训练回归模型
model = RegressionModel(9)
model = model.double()
optimizer = Adam(model.parameters(), lr=1)

num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = nn.MSELoss()(y_pred.squeeze(), Y_train)
    # 添加SCAD惩罚项
    penalty = torch.tensor((0))
    for param in model.parameters():
        for i in range(9):
            penalty = penalty+scad(torch.abs(param[0,i]))
    total_loss = loss + penalty
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Penalty: {penalty.item()}")


from matplotlib import pyplot as plt
y_pre = model(X_test)
def plot_predictions(predictions, actual_values):
    predictions = predictions.detach().numpy().flatten()
    actual_values = actual_values.detach().numpy().flatten()
    plt.plot(predictions, color='blue', label='Predictions')
    plt.plot(actual_values, color='red', label='Actual Values')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()
plot_predictions(y_pre, y_test)

#模型评估
def evaluate_regression(y_true, y_pred):
    y_true = y_true.detach().numpy()
    y_pred = y_pred.squeeze(1).detach().numpy()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    correlation_matrix = np.corrcoef(y_true, y_pred)
    # 提取相关系数
    r = correlation_matrix[0, 1]
    return rmse, mse, mae, r

rmse, mse, mae, r = evaluate_regression(y_test, y_pre)

# 打印结果
print("RMSE:", rmse)
print("MSE:", mse)
print("MAE:", mae)
print("R:", r)