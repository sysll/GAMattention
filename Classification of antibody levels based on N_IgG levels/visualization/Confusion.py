import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 示例真实标签和预测结果
y_true = np.array([1, 0, 0, 1, 1, 0, 1, 0])
y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0])

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# 绘制混淆矩阵
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, [0, 1])
plt.yticks(tick_marks, [0, 1])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 计算召回率（Recall）
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
print("Recall:", recall)

# 计算 F1 分数
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
f1_score = 2 * (precision * recall) / (precision + recall)
print("F1 Score:", f1_score)
