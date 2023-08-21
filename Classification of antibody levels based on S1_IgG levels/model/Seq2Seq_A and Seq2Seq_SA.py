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
    plt.title("(h) Seq2seq_SA")

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

    def forward(self, hidden, encoder_outputs):
        if n == 0:
            energy = torch.matmul(hidden.unsqueeze(0), encoder_outputs.transpose(1, 2))
            attn_weights = self.softmax(energy)  # [batch, 1, seq_len]
            context = torch.matmul(attn_weights, encoder_outputs)  # [batch, 1, hidden]
            return context, attn_weights
        if n == 1:
            energy = torch.matmul(hidden.unsqueeze(0), encoder_outputs.transpose(1, 2))  # [1,1,4][batch, 1, seqlen]
            energy = energy.detach()
            energy = energy.squeeze(1).squeeze(0)  # [4]
            energy = torch.nn.Sigmoid()(energy)
            encoder_outputs = encoder_outputs.squeeze(0)
            c = GAM.get_f(energy, encoder_outputs)
            c = c.unsqueeze(0)
            return c, n

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
for i in range(8):     #5
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



torch.save(model, 'trained model.pth')
model = torch.load('trained model.pth')
model = model.to(device)

#数据的读写

data_num = 800
data_encode_out = torch.zeros((data_num, sequence_len, hidden_size))
data_input = torch.zeros((data_num, sequence_len))     #h.T*s不带W
data_target = torch.zeros((data_num, hidden_size))       #c

for a in range(data_num):
    input = X_train[a, :]
    _, _, c, d, con  = model(input.unsqueeze(0).unsqueeze(2))
    #return outputs, attn_weights, encoder_outputs, hidden_output, con1

    e = torch.matmul(d, c.transpose(1, 2))
    data_input[a] = e.squeeze(0).squeeze(1)
    data_target[a] = con.squeeze(0).squeeze(1)
    data_encode_out[a] = c[0]

data_encode_out = data_encode_out.detach()
data_input =data_input.detach()
data_target = data_target.detach()
sigmoid = torch.nn.Sigmoid()
data_input = sigmoid(data_input)

data_input = data_input.to(device)
data_target = data_target.to(device)



for i in range(1,2):
    torch.manual_seed(14)    #13,14
    #广义加性模型的撰写和训练

    k = 3  #广义加性模型的深度   6

    class GAM(nn.Module):
        def __init__(self):
            super(GAM, self).__init__()


        def kernal(self, x, k): #x是hi.T*S是注意力成绩
            out = torch.randn((k, 1))
            for i in range(1, k+1):
                out1 = np.exp(x/(2**i))
                out[i-1, 0] = out1
            return out

        def smooth_fun(self, x, j):  #传入h.t*s
            mid = self.kernal(x, k)
            return torch.matmul(b[:, j], mid)

        def l2_norm(self, lower_limit, upper_limit, num_points=50):
            norm = torch.zeros((sequence_len))
            xs = torch.linspace(lower_limit, upper_limit, num_points)
            for i in range(sequence_len):
                ys = torch.zeros((num_points))
                for n in range(num_points):
                    ys[n] = self.smooth_fun(xs[n], i)
                squared_norm = torch.norm(ys, p=2) ** 2
                no = torch.sqrt(squared_norm)
                norm[i] = self.scad(no)
            n = torch.sum(norm)
            return n


        def scad(self, beta):
            lambda_ = 0.71
            a = 1
            alpha = 0.71
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
        def get_f(self, dot, encode_out):
            y1 = torch.zeros((1, encode_out.shape[1]))
            for j in range(sequence_len):
                y1 = y1+self.smooth_fun(dot[j], j)*encode_out[j, :]
                "得到去拟合cj的向量"
            return y1

    #广义加性模型的训练
    b = torch.randn((k, sequence_len), requires_grad = True)
    lower_limit = 0
    upper_limit = 1

    criterion = nn.MSELoss()
    optimizer = optim.Adam([b], lr=0.01)

    GA = GAM()
    GAM = GA.to(device)
    ave_loss = torch.zeros((800))
    print("广义加性模型开始")


    for fre in range(1):
        for num in range(800):
            y1 = GAM.get_f(data_input[num, :], data_encode_out[num])
            loss = criterion(y1, data_target[num].unsqueeze(0))
            al_loss = loss+GAM.l2_norm(lower_limit, upper_limit, num_points=50)
            ave_loss[num] = loss
            if num == 799:
                print(f"Loss: {torch.mean(ave_loss).item()}, al_loss: {al_loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if fre == 3:
            optimizer = optim.Adam([b], lr=0.001)


    #模型的评价

    num = 200   #提取的样本数，用评价

    predict = torch.zeros((num, 2))
    label = torch.zeros((num))



    print("普通的注意力机制的结果")
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

    # 提取出分数
    evaluation_outputs = torch.softmax(evaluation_outputs, dim=1)
    evaluation_outputs = evaluation_outputs.detach().numpy()
    # evaluation_outputs = np.max(evaluation_outputs, axis=1)
    evaluation_outputs = evaluation_outputs[:, 1]

    y_test = targeted.detach().numpy().reshape(200)
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

    y_pred_prob = evaluation_outputs
    np.save('D:\\Users\\ASUS\\Desktop\\论文\\基于N_lgG的阴阳分类\\结果\\Seq2seq+attention_predict.npy', y_pred_prob)

    # 计算 ROC 曲线的假正率（FPR）和真正率（TPR）
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_prob)

    # 计算 ROC 曲线下面积（AUC）
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    # 计算 P-R 曲线的精确率（Precision）和召回率（Recall）
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)

    # 计算 P-R 曲线下面积（AP）
    average_precision = average_precision_score(y_test, y_pred_prob)

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # 对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # 绘制 P-R 曲线
    plt.figure()
    plt.step(recall, precision, where='post', label='P-R curve (AP = %0.2f)' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()




    print("用广义加性模型代替注意力机制后的结果")

    e = 0
    n=1

    for a in range(num):
        input = X_test[a, :]
        outputs, atten, _, _, c = model(input.unsqueeze(0).unsqueeze(2))
        predict[e] = outputs[0,0,0:2]
        e = e + 1

    with torch.no_grad():  # 关闭梯度计算
        evaluation_outputs = predict  #[200, 2]
        # evaluation_loss = criterion(evaluation_outputs, y_test)
        evaluation_loss = 0
        _, evaluation_predicted = torch.max(evaluation_outputs, 1)  #[200]
        evaluation_accuracy = (evaluation_predicted == targeted).sum().item() / len(targeted)
        print(f"Evaluation - Loss: {evaluation_loss:.4f}, Accuracy: {evaluation_accuracy:.4f}")
    # torch.save(b, 'matrix.pt')

    confusion(targeted, evaluation_predicted)
""""
普通的注意力机制的结果
Evaluation - Loss: 0.3221, Accuracy: 0.5650
[[19 21]
 [66 94]]
用广义加性模型代替注意力机制后的结果
Evaluation - Loss: 0.2968, Accuracy: 0.6000
[[ 18  22]
 [ 58 102]]

进程已结束,退出代码0
"""

#提取出分数
evaluation_outputs = torch.softmax(evaluation_outputs, dim = 1)
evaluation_outputs = evaluation_outputs.detach().numpy()
# evaluation_outputs = np.max(evaluation_outputs, axis=1)
evaluation_outputs = evaluation_outputs[:, 1]

y_test = targeted.detach().numpy().reshape(200)
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
y_pred_prob = evaluation_outputs
np.save('D:\\Users\\ASUS\\Desktop\\论文\\基于N_lgG的阴阳分类\\结果\\Seq2seq_SA_predict.npy', y_pred_prob)
# 计算 ROC 曲线的假正率（FPR）和真正率（TPR）
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_prob)

# 计算 ROC 曲线下面积（AUC）
roc_auc = roc_auc_score(y_test, y_pred_prob)

# 计算 P-R 曲线的精确率（Precision）和召回率（Recall）
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)

# 计算 P-R 曲线下面积（AP）
average_precision = average_precision_score(y_test, y_pred_prob)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # 对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 绘制 P-R 曲线
plt.figure()
plt.step(recall, precision, where='post', label='P-R curve (AP = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()