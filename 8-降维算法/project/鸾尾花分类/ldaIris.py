import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 自定义列名,鸢尾花
feature_dict = {i:label for i, label in zip(range(4), ('sepal length in cm',
                                                       'sepal width in cm',
                                                       'petal length in cm',
                                                       'petal width in cm',))}
# 读取数据
df = pd.read_csv(filepath_or_buffer='E:\\machineLearning\\kaggle\\dataset\\iris.data', header=None, sep=',',)
df.columns = [l for i, l in sorted(feature_dict.items())] + ['class label']
df.head()

# 四个特征已是数值，但需要转换一下标签
X = df[['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm']].values
y = df['class label'].values
enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1

# 求均值
np.set_printoptions(precision=4) # 4位小数点
mean_vectors = []
for cl in range(1, 4):
    mean_vectors.append(np.mean(X[y == cl], axis=0))
    print('均值类别%s:%s\n' % (cl, mean_vectors[cl-1]))

# 求类内散布矩阵
S_W = np.zeros((4, 4))
for cl, mv in zip(range(1, 4), mean_vectors):
    class_sc_mat = np.zeros((4, 4))
    for row in X[y == cl]:
        row, mv = row.reshape(4, 1), mv.reshape(4, 1)
        class_sc_mat += (row - mv).dot((row - mv).T)
    S_W += class_sc_mat
print('类内散布矩阵：\n', S_W)

# 求类间散布矩阵
overall_mean = np.mean(X, axis=0) # 全局均值
S_B = np.zeros((4, 4))
for i, mean_vec in enumerate(mean_vectors):
    n = X[y == i + 1,:].shape[0]
    mean_vec = mean_vec.reshape(4, 1)
    overall_mean = overall_mean.reshape(4, 1)
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
print('类间散布矩阵：\n', S_B)

# 求解特征向量
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(4, 1)
    print('\n特征向量{}:\n{}'.format(i + 1, eigvec_sc.real))
    print('特征值{:}:{:.2e}'.format(i + 1, eig_vals[i].real))

# 按特征值大小排序
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
print('特征值排序结果:\n')
for i in eig_pairs:
    print(i[0])

print('特征值占总体百分比:\n')
eigv_sum = sum(eig_vals)
for i, j in enumerate(eig_pairs):
    print('特征值{0:}:{1:.2%}'.format(i + 1, (j[0] / eigv_sum).real))

# 选择将数据降到二维，只选择特征1，2所对应特征向量
W = np.hstack((eig_pairs[0][1].reshape(4, 1), eig_pairs[1][1].reshape(4, 1)))
print('矩阵W：\n', W.real) # 即最终所需投影方向

# 带入原数据，即为降维结果
X_lda = X.dot(W)
X_lda.shape

# 对于较大数据集，采用sklearn来完成降维，直接得到一步步计算矩阵的结果
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
sklearn_lda = LDA(n_components=2)
X_lda_sklearn = sklearn_lda.fit_transform(X, y)
X_lda_sklearn.shape