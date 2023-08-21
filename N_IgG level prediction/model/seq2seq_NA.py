import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
input_size = 1
hidden_size = 40        #40周围效果较好
output_size = 1
num_layers = 1
nnum_layers = 1
batch_size = 1  # 测试数据的batch size
sequence_len = 9  # 测试数据序列的长度
n = 0   #先用普通的注意力机制来训练



torch.manual_seed(44)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, nnum_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = nnum_layers
        self.lstm = nn.LSTM(input_size, hidden_size, nnum_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        output, (hidden, cell) = self.lstm(x, (h0, c0))
        return output, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

    def forward(self, hidden, encoder_outputs):
            energy = torch.matmul(hidden.unsqueeze(0), encoder_outputs.transpose(1, 2))
            energy = self.relu(energy)
            attn_weights = self.softmax(energy)
            context = torch.matmul(attn_weights, encoder_outputs)  # [batch, 1, hidden]
            return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size+1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attention = Attention(hidden_size)

    def forward(self, x, hidden, encoder_outputs, cell):
        context, attn_weights = self.attention(hidden[-1], encoder_outputs)
        con = context
        x = torch.cat((context, x), -1)
        output, (hidden, cell) = self.lstm(x, (hidden[-1].unsqueeze(0), cell[-1].unsqueeze(0)))
        output = self.fc(output)
        return output, hidden, cell, attn_weights, con


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoder_outputs, hidden, cell = self.encoder(x)
        decoder_input = torch.zeros(x.size(0), 1, output_size)
        hidden_output = hidden
        con1 = torch.zeros((1, 1, hidden_size))
        outputs = torch.zeros((1, 1, 1))
        for i in range(1):
            output, hidden, cell, attn_weights, con = self.decoder(decoder_input, hidden, encoder_outputs, cell)
            con1 = con
            outputs[:, i, :] = output.squeeze(0)

        return outputs, attn_weights, encoder_outputs, hidden_output, con1



#训练模型
# 实例化模型和优化器
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
encoder = Encoder(input_size, hidden_size, nnum_layers)
decoder = Decoder(hidden_size, output_size, num_layers)
model = Seq2Seq(encoder, decoder)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
#
torch.manual_seed(42)
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
#
X_train = torch.Tensor(X_train).to(device)
X_test = torch.Tensor(X_test).to(device)
Y_train = torch.Tensor(Y_train).to(device)
y_test = torch.Tensor(Y_test).to(device)
model = model.to(device)
# # 训练模型
loss_sum = torch.zeros((800))
for i in range(2): #9，8，7
    for a in range(800):
        input = X_train[a, :]
        target = Y_train[a]
        target = torch.tensor([target])
        optimizer.zero_grad()
        outputs, atten,_,_,_ = model(input.unsqueeze(0).unsqueeze(2))
        loss = criterion(outputs, target.unsqueeze(0).unsqueeze(2))
        loss.backward()
        optimizer.step()
        loss_sum[a] = loss
    print("第"+str(i)+"次的误差平均数是"+str(torch.mean(loss_sum)))



#模型的评价

num = 200   #提取的样本数，用评价

predict = torch.zeros((num))
label = torch.zeros((num))

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

print("普通的注意力机制的结果")
e = 0   #用于给predict赋值
n = 0   #普通注意力机制
for a in range(num):
    input = X_test[a, :]
    target = Y_test[a]
    target = torch.tensor([target])
    outputs, atten, _, _, c = model(input.unsqueeze(0).unsqueeze(2))
    predict[e] = outputs[0,0,0]
    label[e] = target.flatten()
    e = e + 1

mse = calculate_mse(predict, label)
rmse = calculate_rmse(predict, label)
MAE = calculate_mae(predict, label)
R = calculate_r_squared(predict, label)

print("RMSE: {:.5f}".format(round(rmse.item(), 5)))
print("MSE: {:.5f}".format(round(mse.item(), 5)))
print("MAE: {:.5f}".format(round(MAE.item(), 5)))
print("R-squared: {:.5f}".format(round(R, 5)))