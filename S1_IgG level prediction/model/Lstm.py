import torch
import torch.nn as nn
import numpy as np
import pandas as pd
torch.manual_seed(5)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 只使用最后一个时间步的输出
        return out

# 定义超参数
input_size = 9  # 输入特征数量
hidden_size = 20  # LSTM隐藏层大小
num_layers =2   # LSTM层数
output_size = 1  # 输出大小
num_epochs = 6  # 训练轮数
learning_rate = 0.01  # 学习率

# 创建模型实例
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
train_path = "D:\\Users\\ASUS\\Desktop\\论文\\抗体水平检预测\用于调参的模型\\used for train.xlsx"
test_path = "D:\\Users\\ASUS\\Desktop\\论文\\抗体水平检预测\\用于调参的模型\\used for test.xlsx"
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

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
Y_train = torch.Tensor(Y_train)
y_test = torch.Tensor(Y_test)




total_loss = 0  # 总的loss
count = 0  # 计数器，用于每个epoch计算平均值

for epoch in range(num_epochs):
    for a in range(800):
        input = X_train[a, :]
        target = Y_train[a]
        target = torch.tensor([target])

        output = model(input.unsqueeze(0).unsqueeze(1))
        loss = criterion(output, target.unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / count
    count = 0  # 重置计数器
    total_loss = 0  # 重置总的loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
# 测试模型
def calculate_mse(prediction, target):
    mse_loss = torch.nn.MSELoss()
    mse = mse_loss(prediction, target)

    return mse

def calculate_rmse(prediction, target):
    mse_loss = torch.nn.MSELoss()
    mse = mse_loss(prediction, target)
    rmse = torch.sqrt(mse)
    return rmse

def calculate_mae(prediction, target):
    mae_loss = torch.nn.L1Loss()
    mae = mae_loss(prediction, target)
    return mae



def calculate_r_squared(predicted_values, actual_values):
    y_pred = predicted_values.detach().numpy()
    y_true = actual_values.detach().numpy()
    correlation_matrix = np.corrcoef(y_true, y_pred)
    # 提取相关系数
    r = correlation_matrix[0, 1]
    return r
pred = torch.zeros((200))
with torch.no_grad():
    for a in range(200):
        input = X_test[a, :]
        output = model(input.unsqueeze(0).unsqueeze(1))
        part_pred = output.squeeze(0)
        pred[a] = part_pred
    prediction = pred
    target = y_test

    mse = calculate_mse(prediction, target)
    rmse = calculate_rmse(prediction, target)
    mae = calculate_mae(prediction, target)
    r_squared = calculate_r_squared(prediction, target)

    print("MSE:", mse.item())
    print("RMSE:", rmse.item())
    print("MAE:", mae.item())
    print("R-squared:", r_squared)