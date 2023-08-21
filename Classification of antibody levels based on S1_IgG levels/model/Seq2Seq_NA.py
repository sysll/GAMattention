import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
input_size = 1
hidden_size = 40        #40周围效果较好
output_size = 2
num_layers = 1
nnum_layers = 1
batch_size = 1  # 测试数据的batch size
sequence_len = 9  # 测试数据序列的长度
n = 0   #先用普通的注意力机制来训练

from sklearn.metrics import precision_score
def confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    # 计算类别1的Recall和F1-score
    TP1 = cm[1, 1]
    FN1 = cm[1, 0]
    recall1 = TP1 / (TP1 + FN1)
    precision1 = precision_score(y_true, y_pred)
    f1_score1 = 2 * (precision1 * recall1) / (precision1 + recall1)
    print("对类别1的Recall:", recall1)
    print("对类别1的F1-score:", f1_score1)

    # 计算类别0的Recall和F1-score
    TP0 = cm[0, 0]
    FN0 = cm[0, 1]
    recall0 = TP0 / (TP0 + FN0)
    precision0 = precision_score(y_true, y_pred, pos_label=0)
    f1_score0 = 2 * (precision0 * recall0) / (precision0 + recall0)
    print("对类别0的Recall:", recall0)
    print("对类别0的F1-score:", f1_score0)

    # 计算每个类别的总数
    class_totals = np.sum(cm, axis=1)

    # 计算每个矩阵格的比例值
    cm_proportions = cm / class_totals[:, np.newaxis]

    # 设置标签和标题
    labels = ['0', '1']
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_proportions, annot=True, fmt=".4f", cmap='Blues', vmin=0.35, vmax=0.6)

    # 设置坐标轴标签和标题
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("(d) Seq2seq_NA")

    # 设置坐标轴刻度标签
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks + 0.5, labels)
    plt.yticks(tick_marks + 0.5, labels)

    # 显示图形
    plt.show()

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
            attn_weights = self.softmax(energy)  # [batch, 1, seq_len]
            context = torch.matmul(attn_weights, encoder_outputs)  # [batch, 1, hidden]
            return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size+2, hidden_size, num_layers, batch_first=True)
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
        outputs = torch.zeros((1, 1, 2))
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
X_train = torch.Tensor(X_train).to(device)      #[800,9]
X_test = torch.Tensor(X_test).to(device)        #[200,9]


Y_train = torch.Tensor(Y_train).long()      #torch.Size([800])
y_test = torch.Tensor(Y_test).long()      #torch.Size([200])
targeted = y_test

num_classes = 2  # 二分类问题，有两个类别

# 将 Y_train 转换为 one-hot 编码，并保存在 Y_train 中# 将 Y_test 转换为 one-hot 编码，并保存在 Y_test 中
Y_train = F.one_hot(Y_train, num_classes=num_classes).to(device)
y_test = F.one_hot(y_test, num_classes=num_classes).to(device)
Y_train = Y_train.float()
y_test = y_test.float()

model = model.to(device)

# # 训练模型
loss_sum = torch.zeros((800))
for i in range(5):     #5
    for a in range(800):
        input = X_train[a, :]
        target = Y_train[a]
        optimizer.zero_grad()
        outputs, atten,_,_,_ = model(input.unsqueeze(0).unsqueeze(2))
        loss = criterion(outputs, target.unsqueeze(0).unsqueeze(1))
        loss.backward()
        optimizer.step()
        loss_sum[a] = loss
    print("第"+str(i)+"次的误差平均数是"+str(torch.mean(loss_sum)))


#模型的评价

num = 200   #提取的样本数，用评价

predict = torch.zeros((num, 2))
label = torch.zeros((num))



e = 0   #用于给predict赋值
n = 0   #普通注意力机制
for a in range(num):
    input = X_test[a, :]
    outputs, atten, _, _, c = model(input.unsqueeze(0).unsqueeze(2))
    predict[e] = outputs[0,0,0:2]
    e = e + 1

with torch.no_grad():  # 关闭梯度计算
        evaluation_outputs = predict  #[200, 2]
        evaluation_loss = criterion(evaluation_outputs, y_test)
        _, evaluation_predicted = torch.max(evaluation_outputs, 1)  #[200]
        evaluation_accuracy = (evaluation_predicted == targeted).sum().item() / len(targeted)
        print(f"Evaluation - Loss: {evaluation_loss:.4f}, Accuracy: {evaluation_accuracy:.4f}")

    # 计算混淆矩阵
confusion(targeted, evaluation_predicted)

predict = torch.softmax(predict, dim = 1)
score = predict[:, 1].detach().numpy()
np.save('D:\\Users\\ASUS\\Desktop\\论文\\基于N_lgG的阴阳分类\\结果\\Seq2seq_NA.npy', score)
