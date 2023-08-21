import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
train_path = "N_used for train.xlsx"
test_path = "N_used for test.xlsx"

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
    plt.title("(c) AdaBoost")

    # 设置坐标轴刻度标签
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks + 0.5, labels)
    plt.yticks(tick_marks + 0.5, labels)

    # 显示图形
    plt.show()
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
base_estimator = DecisionTreeClassifier(max_depth=3)  # Example of a base estimator
adaboost = AdaBoostClassifier()
adaboost.fit(X_train, Y_train)
predictions = adaboost.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy)
confusion(Y_test, predictions)


y_pred_prob = adaboost.predict_proba(X_test)[:, 1]
np.save('D:\\Users\\ASUS\\Desktop\\论文\\基于N_lgG的阴阳分类\\结果\\AdaBoost_predict.npy', y_pred_prob)
