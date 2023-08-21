import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error \
    , mean_absolute_percentage_error, r2_score
from itertools import chain
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt


seed = 5

train_path = "D:\\Users\\ASUS\\Desktop\\论文\\抗体水平检预测\\特征选择代码\\processed train_data without division.xlsx"
test_path = "D:\\Users\\ASUS\\Desktop\\论文\\抗体水平检预测\\特征选择代码\\processed test_data without division.xlsx"
#所有特征的导入
feature=['性别', '高血压(0=无，1=有)', '糖尿病(0=无，1=有)', '年龄','血_血红蛋白', '血_淋巴细胞(#)',
'血_淋巴细胞(%)', '血_平均血红蛋白含量', '血_平均血红蛋白浓度', '血_平均RBC体积', '血_单核细胞(#)', '血_单核细胞(%)',
'血_中性粒细胞(#)', '血_中性粒细胞(%)', '血_血小板计数', '血_红细胞计数', '血_嗜碱细胞(#)', '血_嗜碱细胞(%)', '血_嗜酸细胞(#)',
'血_嗜酸细胞(%)', '血_红细胞压积', '血_白细胞计数', '血_RBC分布宽度CV', '血_RBC分布宽度SD', '血_平均PLT体积', '血_PLT分布宽度',
'血_大血小板比率', '血_血小板压积', '血_钾', '血_钙', '血_钠', '血_氯', '血_谷草转氨酶', '血_乳酸脱氢酶', '血_LDH*0.9', '血_谷丙转氨酶',
'血_总胆固醇', '血_白蛋白', '血_碱性磷酸酶', '血_γ-谷氨酰转肽酶', '血_总胆红素', '血_总蛋白', '血_白/球比值', '血_球蛋白', '血_TP*0.75',
'血_直接胆红素', '血_TBIL*0.8', '血_间接胆红素', '血_肌酐', '血_尿素', '血_尿酸', '血_碳酸氢根', '血_eGFR(基于CKD-EPI方程)', '血_校正钙']
df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)

Y_predict = []
Y_true = []

for i in range(1):
        X_train = df_train[feature]
        Y_train = df_train['S1_IgG']

        X_test = df_test[feature]
        Y_test = df_test['S1_IgG']

        X_train = np.array(X_train)
        X_test = np.array(X_test)

        Y_train = np.array(Y_train)
        Y_test = np.array(Y_test)
        #用原始参数
        CatBoost_Model = CatBoostRegressor()
        CatBoost_Model.fit(X_train, Y_train, eval_set=(X_test, Y_test), verbose=True)
        part_predict = CatBoost_Model.predict(X_test)
        Y_predict.append(part_predict)
        Y_true.append(Y_test)

        Y_predict = np.concatenate(Y_predict)
        Y_true = np.concatenate(Y_true)

        plt.figure(figsize=(50, 4))

        print('MSE: {}'.format(mean_squared_error(Y_true, Y_predict)))
        print('RMSE: {}'.format(np.sqrt(mean_squared_error(Y_true, Y_predict))))
        print('MAE: {}'.format(mean_absolute_error(Y_true, Y_predict)))
        print('MAPE: {}'.format(mean_absolute_percentage_error(Y_true, Y_predict)))
        print('R: {}'.format(np.corrcoef(Y_true, Y_predict)[0, 1]))

        # plt.title('S1_IgG Antibody levels prediction based on CatBoost')
        # plt.plot(Y_true, "blue")
        # plt.plot(Y_predict, "red")
        # plt.legend(['True', 'Predict'], loc='best')
        # plt.show()
        # # 特征提取
        importance = CatBoost_Model.feature_importances_
        indices = np.argsort(importance)[::-1]  # 降序排列特征重要性的索引

        # 打印特征重要性
        print("Feature Importance:")
        for f in range(X_train.shape[1]):
                print(f"{feature[indices[f]]}: {importance[indices[f]]}")