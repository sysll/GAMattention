import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc,average_precision_score

# Load model predictions and true labels
model1 = np.load('D:\\Users\\ASUS\Desktop\\论文\\基于S1_lgG的阴阳分类\\结果\\adaBoost_predict.npy')
model2 = np.load('D:\\Users\\ASUS\Desktop\\论文\\基于S1_lgG的阴阳分类\\结果\\CatBoost_predict.npy')
model3 = np.load('D:\\Users\\ASUS\Desktop\\论文\\基于S1_lgG的阴阳分类\\结果\\Light_predict.npy')
model4 = np.load('D:\\Users\\ASUS\Desktop\\论文\\基于S1_lgG的阴阳分类\\结果\\Seq2seq+attention_predict.npy')
model5 = np.load('D:\\Users\\ASUS\Desktop\\论文\\基于S1_lgG的阴阳分类\\结果\\Seq2seq_SA_predict.npy')
model6 = np.load('D:\\Users\\ASUS\Desktop\\论文\\基于S1_lgG的阴阳分类\\结果\\Seq2seq_NA.npy')
model7 = np.load('D:\\Users\\ASUS\Desktop\\论文\\基于S1_lgG的阴阳分类\\结果\\GAM_predict.npy')
model8 = np.load('D:\\Users\\ASUS\Desktop\\论文\\基于S1_lgG的阴阳分类\\结果\\LSTM.npy')

y_true = np.load('D:\\Users\\ASUS\\Desktop\\论文\\基于S1_lgG的阴阳分类\\结果\\CatBoost_test.npy')

# Calculate ROC curve and AUC for each model
fpr_model1, tpr_model1, _ = roc_curve(y_true, model1)
fpr_model2, tpr_model2, _ = roc_curve(y_true, model2)
fpr_model3, tpr_model3, _ = roc_curve(y_true, model3)
fpr_model4, tpr_model4, _ = roc_curve(y_true, model4)
fpr_model5, tpr_model5, _ = roc_curve(y_true, model5)
fpr_model6, tpr_model6, _ = roc_curve(y_true, model6)
fpr_model7, tpr_model7, _ = roc_curve(y_true, model7)
fpr_model8, tpr_model8, _ = roc_curve(y_true, model8)

auc_model1 = roc_auc_score(y_true, model1)
auc_model2 = roc_auc_score(y_true, model2)
auc_model3 = roc_auc_score(y_true, model3)
auc_model4 = roc_auc_score(y_true, model4)
auc_model5 = roc_auc_score(y_true, model5)
auc_model6 = roc_auc_score(y_true, model6)
auc_model7 = roc_auc_score(y_true, model7)
auc_model8 = roc_auc_score(y_true, model8)

# Calculate Precision-Recall curve and AUC for each model
precision_model1, recall_model1, _ = precision_recall_curve(y_true, model1)
precision_model2, recall_model2, _ = precision_recall_curve(y_true, model2)
precision_model3, recall_model3, _ = precision_recall_curve(y_true, model3)
precision_model4, recall_model4, _ = precision_recall_curve(y_true, model4)
precision_model5, recall_model5, _ = precision_recall_curve(y_true, model5)
precision_model6, recall_model6, _ = precision_recall_curve(y_true, model6)
precision_model7, recall_model7, _ = precision_recall_curve(y_true, model7)
precision_model8, recall_model8, _ = precision_recall_curve(y_true, model8)

ap_model1 = average_precision_score(y_true, model1)
ap_model2 = average_precision_score(y_true, model2)
ap_model3 = average_precision_score(y_true, model3)
ap_model4 = average_precision_score(y_true, model4)
ap_model5 = average_precision_score(y_true, model5)
ap_model6 = average_precision_score(y_true, model6)
ap_model7 = average_precision_score(y_true, model7)
ap_model8 = average_precision_score(y_true, model8)

# Create figure and axes for ROC and P-R curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot ROC curves
ax1.plot(fpr_model1, tpr_model1, color=(137/255, 215/255, 188/255), label='AdaBoost (AUC = {:.4f})'.format(auc_model1))
ax1.plot(fpr_model2, tpr_model2, color=(153/255, 195/255, 217/255), label='CatBoost (AUC = {:.4f})'.format(auc_model2))
ax1.plot(fpr_model3, tpr_model3, color=(61/255, 112/255, 143/255), label='LightGBM (AUC = {:.4f})'.format(auc_model3))
ax1.plot(fpr_model4, tpr_model4, color=(147/255, 148/255, 231/255), label='Seq2seq_A (AUC = {:.4f})'.format(auc_model4))
ax1.plot(fpr_model5, tpr_model5, color=(255/255, 208/255, 111/255), label='Seq2seq_SA (AUC = {:.4f})'.format(auc_model5))
ax1.plot(fpr_model6, tpr_model6, color=(115/255, 186/255, 214/255), label='Seq2seq_NA (AUC = {:.4f})'.format(auc_model6))
ax1.plot(fpr_model7, tpr_model7, color=(200/255, 65/255, 67/255), label='LSTM_WAA (AUC = {:.4f})'.format(auc_model7))
ax1.plot(fpr_model8, tpr_model8, color=(229/255, 133/255, 93/255), label='GAM (AUC = {:.4f})'.format(auc_model8))
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('(a) ROC Curve')
ax1.legend(loc='lower right')

# Plot P-R curves
ax2.plot(recall_model1, precision_model1, color=(137/255, 215/255, 188/255), label='AdaBoost (AP = {:.4f})'.format(ap_model1))
ax2.plot(recall_model2, precision_model2, color=(153/255, 195/255, 217/255), label='CatBoost (AP = {:.4f})'.format(ap_model2))
ax2.plot(recall_model3, precision_model3, color=(61/255, 112/255, 143/255), label='LightGBM (AP = {:.4f})'.format(ap_model3))
ax2.plot(recall_model4, precision_model4, color=(147/255, 148/255, 231/255), label='Seq2seq_A (AP = {:.4f})'.format(ap_model4))
ax2.plot(recall_model5, precision_model5, color=(255/255, 208/255, 111/255), label='Seq2seq_SA (AP = {:.4f})'.format(ap_model5))
ax2.plot(recall_model6, precision_model6, color=(115/255, 186/255, 214/255), label='Seq2seq_NA (AP = {:.4f})'.format(ap_model6))
ax2.plot(recall_model7, precision_model7, color=(200/255, 65/255, 67/255), label='LSTM_WAA (AP = {:.4f})'.format(ap_model7))
ax2.plot(recall_model8, precision_model8, color=(229/255, 133/255, 93/255), label='GAM (AP = {:.4f})'.format(ap_model8))

ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('(b) Precision-Recall Curve')
ax2.legend(loc='lower right')

# Show the plots
plt.tight_layout()
plt.show()