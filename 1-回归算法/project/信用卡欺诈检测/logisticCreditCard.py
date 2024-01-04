import itertools

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split


#读取数据
data = pd.read_csv("E:\machineLearning\kaggle\dataset\creditCard.csv")
data.head()


#数据比例
# count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
# count_classes.plot(kind = 'bar')
# plt.title("Fraud class histogram")
# plt.xlabel("Class")
# plt.ylabel("Frequency")


#特征标准化
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)
data.head()


#下采样
#特征，不包含标签
X = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']
number_records_fraud = len(data[data.Class == 1])
#所有异常样本索引
fraud_indices = np.array(data[data.Class == 1].index)
#所有正常样本索引
normal_indices = data[data.Class == 0].index
#随机采样指定个数正常样本，取其索引
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)
#正常和异常样本索引
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
#根据索引下采样
under_sample_data = data.iloc[under_sample_indices,:]
X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'Class']
#打印比例
print("normal:", len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
print("abnormal:", len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
print("all samples:", len(under_sample_data))


#数据集切分
#整个划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print("train amount:", len(X_train))
print("test amount:", len(X_test))
print("all amount:", len(X_train) + len(X_test))
#下采样划分
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = (
    train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=0))
print("low train amount:", len(X_train_undersample))
print("low test amount:", len(X_test_undersample))
print("low all amount:", len(X_train_undersample) + len(X_test_undersample))


#正则化惩罚力度的调参实验
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
def priting_Kfold_scores(x_train_data, y_train_data):
    fold = KFold(5, shuffle=False)
    c_param_range = [0.01, 0.1, 1, 10, 100]
    #结果展示表格
    results_table = pd.DataFrame(index=range(len(c_param_range), 2),
                                 columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range
    #K交叉验证，得到索引集合：训练集=indices[0]；验证集=indices[1]
    j = 0
    for c_param in c_param_range:
        print('----------------------------')
        print('正则化惩罚力度：', c_param)
        print('----------------------------')
        print('')
        recall_accs = []
        for iteration, indices in enumerate(fold.split(y_train_data), start=1):
            lr = LogisticRegression(C = c_param, penalty='l1', solver='liblinear')
            #训练模型
            lr.fit(x_train_data.iloc[indices[0],:].values, y_train_data.iloc[indices[0],:].values.ravel())
            #验证集
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)
            #评估
            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values, y_pred_undersample)
            #保存结果
            recall_accs.append(recall_acc)
            print('Iteration', iteration, ':召回率 = ', recall_acc)
        #计算平均
        results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('平均召回率：', np.mean(recall_accs))
        print('')
    #最好参数
    best_c = results_table.loc[results_table['Mean recall score'].astype('float32').idxmax()]['C_parameter']
    print('*******************************')
    print('效果最好模型所选参数 = ', best_c)
    print('*******************************')
    return best_c
best_c = priting_Kfold_scores(X_train_undersample, y_train_undersample)


#混淆矩阵
from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes, title = 'Confusion matrix', cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color = "white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#处理过的数据集
lr = LogisticRegression(C=best_c, penalty='l1', solver='liblinear')
lr.fit(X_train_undersample.values, y_train_undersample.values.ravel())
y_pred_undersample = lr.predict(X_test_undersample.values)
#计算所需值
cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
np.set_printoptions(precision=2)
print("召回率：", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
#绘制
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

#原始数据集
lr = LogisticRegression(C=best_c, penalty='l1', solver='liblinear')
lr.fit(X_train_undersample.values, y_train_undersample.values.ravel())
y_pred = lr.predict(X_test.values)
#计算所需值
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print("召回率 in testing dataset：", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
#绘制
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()


#阈值分类
lr =LogisticRegression(C = 0.01, penalty= 'l1', solver='liblinear')
lr.fit(X_train_undersample.values, y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)
#指定不同阈值
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
plt.figure(figsize=(10, 10))
j = 1
#混淆矩阵展示
for i in thresholds:
    y_test_predictions_high_recall = y_pred_undersample_proba[:,1] > i #比较概率和阈值
    plt.subplot(3, 3, j)
    j += 1
    cnf_matrix = confusion_matrix(y_test_undersample, y_test_predictions_high_recall)
    np.set_printoptions(precision=2)
    print("Recall metric in the testing dataset:",
          cnf_matrix[1, 1,] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
    class_names = [0, 1]
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >= %s' %i)



#过采样方案
from imblearn.over_sampling import SMOTE
oversampler = SMOTE(random_state=0)
os_features, os_lables = oversampler.fit_resample(X_train, y_train)
#训练集
os_features = pd.DataFrame(os_features)
os_lables = pd.DataFrame(os_lables)
best_c = priting_Kfold_scores(os_features, os_lables)

#测试结果的混淆矩阵
lr = LogisticRegression(C = best_c, penalty='l1' ,solver='liblinear')
lr.fit(os_features.values, os_lables.values.ravel())
y_pred = lr.predict(X_test.values)

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print("recall matric in the testing dataset:", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()