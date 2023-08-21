import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# 创建示例数据
train_path = "S1_used for train.xlsx"
test_path = "S1_used for test.xlsx"


feature = ['年龄', '血_血小板压积', '血_RBC分布宽度SD', '血_RBC分布宽度CV', '血_碳酸氢根', '血_eGFR(基于CKD-EPI方程)', '血_红细胞压积', '血_球蛋白', '血_红细胞计数']

df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)

X_train = df_train[feature]
y_train = df_train['S1_IgG']

X_test = df_test[feature]
y_test = df_test['S1_IgG']

# 创建LightGBM分类器
clf = lgb.LGBMClassifier(max_depth=5, learning_rate=0.1, n_estimators=300, num_leaves=30,
                       subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1)

# 在训练集上拟合模型
clf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = clf.predict(X_test)

y_pred_prob = clf.predict_proba(X_test)[:, 1]
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)
np.save('D:\\Users\\ASUS\\Desktop\\论文\\基于S1_lgG的阴阳分类\\结果\\Light_predict.npy', y_pred_prob)

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
    sns.heatmap(cm_proportions, annot=True, fmt=".4f", cmap='Blues', vmin=0.33, vmax=0.6)

    # 设置坐标轴标签和标题
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("(a) LightGBM")

    # 设置坐标轴刻度标签
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks + 0.5, labels)
    plt.yticks(tick_marks + 0.5, labels)

    # 显示图形
    plt.show()
confusion(y_test, y_pred)