import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns


#1. 读取数据集
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

#2. 使用PCA降维，数据划分
#降到150维
pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)
#划分数据集
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=40)

#3. SVM训练，使用网络搜索选择C和gamma参数
param_grid = {'svc__C': [1, 5, 10], 'svc__gamma': [0.0001, 0.0005, 0.001]}
grid = GridSearchCV(model, param_grid)
#%time grid.fit(Xtrain, ytrain)
grid.fit(Xtrain, ytrain)
print(grid.best_params_)

#4. 结果预测
model = grid.best_estimator_
yfit = model.predict(Xtest)
yfit.shape
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1], color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
#各项具体评估指标
from sklearn.metrics import classification_report
print(classification_report(ytest, yfit, target_names=faces.target_names))

#混淆矩阵观察
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');
