from sklearn.metrics import precision_score
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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
    plt.title("(e) LSTM_WAA")

    # 设置坐标轴刻度标签
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks + 0.5, labels)
    plt.yticks(tick_marks + 0.5, labels)

    # 显示图形
    plt.show()
class AttentionLSTMClassifier(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AttentionLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Attention layer
        self.attention = nn.Linear(hidden_size, 1)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Input x shape: (batch_size, seqlen, hidden_size)
        # LSTM layer
        lstm_output, _ = self.lstm(x)   #[1 ,9 ,16]

        # Attention layer
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)   #[1, 9, 1]，各个seq的注意力分数
        attention_output = torch.sum(attention_weights * lstm_output, dim=1)    #[1, 9, 1]*[1 ,9 ,16]

        # Fully connected layer for classification
        output = self.fc(attention_output)

        return torch.softmax(output,dim = 1)


# Example usage:
batch_size = 1
seqlen = 9
hidden_size = 16
input_size = 1
output_size = 2

import torch.optim as optim

# # Example data and labels (assuming binary classification)
# data = torch.randn(800, seqlen, 1)  # Assuming you have 800 samples
# labels = torch.randint(0, 2, (800,))  # Binary labels (0 or 1)
#输入数据
torch.manual_seed(42)
train_path = "S1_used for train.xlsx"
test_path = "S1_used for test.xlsx"
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
#
X_train = torch.Tensor(X_train)     #[800,9]
X_test = torch.Tensor(X_test)        #[200,9]
data = X_train.unsqueeze(2)
X_test = X_test.unsqueeze(2)

Y_train = torch.Tensor(Y_train)     #torch.Size([800])
y_test = torch.Tensor(Y_test)      #torch.Size([200])
labels = Y_train

# Initialize the model
model = AttentionLSTMClassifier(hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 42
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i in range(len(data)):
        input_data = data[i:i+1]  # Get one sample at a time
        label = labels[i:i+1]
        optimizer.zero_grad()
        label = label.to(torch.long)
        # Forward pass
        output = model(input_data)
        # Calculate loss
        loss = criterion(output, label)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(data)}")

# Example usage after training
# Example usage after training
model.eval()
correct_predictions = 0
predicted = torch.tensor((200))
# Evaluate on the test set (200 samples)
# Example usage after training
model.eval()
predicted = []
score = torch.zeros((200, 2))
# Evaluate on the test set (200 samples)
with torch.no_grad():
    for i in range(len(X_test)):
        test_input = X_test[i:i+1]  # Get one sample at a time
        label = y_test[i:i+1]
        label = label.to(torch.long)

        output = model(test_input)
        _, predicted_class = torch.max(output, 1)
        score[i] = output
        predicted.append(predicted_class.item())  # 将预测结果转换为Python标量并添加到predicted列表中

# Convert the predicted list to a numpy array
predicted = np.array(predicted)

# Calculate accuracy
correct_predictions = (predicted == Y_test).sum()
accuracy = correct_predictions / len(Y_test) * 100
print(f"Accuracy on the test set: {accuracy:.2f}%")

confusion(y_test, predicted)
np.save('D:\\Users\\ASUS\\Desktop\\论文\\基于S1_lgG的阴阳分类\\结果\\LSTM.npy', score[:, 1])