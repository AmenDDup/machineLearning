import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. 读取数据
df = pd.read_csv('E:\\machineLearning\\kaggle\\dataset\\iris.data')
df.columns = ['sepal_len', 'speal_wid', 'petal_len', 'petal_wid', 'class']
df.head()

# 2. 将数据分为特征和标签
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values
label_dict = {1: 'Iris-Setosa', 2: 'Iris-Versicolor', 3: 'Iris-Virgnica'}
feature_dict = {0: 'sepal length [cm]', 1: 'sepal width [cm]', 2: 'petal length [cm]', 3: 'petal length [cm]'}
plt.figure(figsize=(8, 6))
for cnt in range(4):
    plt.subplot(2, 2, cnt + 1)
    for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virgnica'):
        plt.hist(X[y == lab, cnt], label=lab, bins=10, alpha=0.3,)
    plt.xlabel(feature_dict[cnt])
    plt.legend(loc='upper right', fancybox=True, fontsize=8)
plt.tight_layout()
plt.show()

# 3. 数据标准化处理
X_std = StandardScaler().fit_transform(X)

# 4. 计算协方差矩阵
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
print('协方差矩阵 \n%s' %cov_mat)

# 5. 求特征值和特征向量
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('特征值 \n%s' %eig_vecs)
print('\n特征向量 \n%s' %eig_vals)

# 6. 特征值大小排序
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
print('特征值排序结果:\n')
for i in eig_pairs:
    print(i[0])

# 7. 计算累加贡献值
tot = sum(eig_vals)
var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
print(var_exp)
cum_var_exp = np.cumsum(var_exp)
cum_var_exp # 发现使用前两个特征对应累计贡献度已到95%，故选择降到二维

# 8. 完成PCA降维
matrix_w = np.hstack((eig_pairs[0][1].reshape(4, 1), eig_pairs[1][1].reshape(4, 1)))
Y = X_std.dot(matrix_w)