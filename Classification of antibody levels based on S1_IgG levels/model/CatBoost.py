import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

# 创建示例数据
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
    plt.title("(b) CatBoost")

    # 设置坐标轴刻度标签
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks + 0.5, labels)
    plt.yticks(tick_marks + 0.5, labels)

    # 显示图形
    plt.show()

train_path = "N_used for train.xlsx"
test_path = "N_used for test.xlsx"

#随机森林特征
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
y_train = df_train['N_IgG']

X_test = df_test[feature]
y_test = df_test['N_IgG']


# 创建CatBoost分类器
clf = CatBoostClassifier()

# 在训练集上拟合模型
clf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:, 1]
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)
np.save('D:\\Users\\ASUS\\Desktop\\论文\\基于N_lgG的阴阳分类\\结果\\CatBoost_predict.npy', y_pred_prob)
np.save('D:\\Users\\ASUS\\Desktop\\论文\\基于N_lgG的阴阳分类\\结果\\CatBoost_test.npy', y_test)

confusion(y_test, y_pred)




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