from sklearn.metrics import precision_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from pygam import LogisticGAM, s
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

y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# Standardize the features (optional but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the Generalized Additive Model (GAM)
n_splines = 15  # Change the number of splines as needed
gam = LogisticGAM(s(0, n_splines=n_splines) + s(1, n_splines=n_splines) + s(2, n_splines=n_splines) +
                  s(3, n_splines=n_splines) + s(4, n_splines=n_splines) + s(5, n_splines=n_splines) +
                  s(6, n_splines=n_splines) + s(7, n_splines=n_splines) + s(8, n_splines=n_splines))

# Fit the model to the training data
gam.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = gam.accuracy(X_test, Y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

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
    plt.title("(f) GAN")

    # 设置坐标轴刻度标签
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks + 0.5, labels)
    plt.yticks(tick_marks + 0.5, labels)

    # 显示图形
    plt.show()

# Get the predictions on the test data
y_pred = gam.predict(X_test)
# Convert the probabilities to binary predictions (0 or 1)
y_pred_binary = (y_pred >= 0.5).astype(int)
confusion(Y_test, y_pred)
y_pred_prob = gam.predict_proba(X_test)
np.save('D:\\Users\\ASUS\\Desktop\\论文\\基于S1_lgG的阴阳分类\\结果\\GAM_predict.npy', y_pred_prob)