import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import fetch_openml

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

warnings.filterwarnings('ignore')
np.random.seed(42)

mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target']
#print(X.shape) # (70000, 784), 28*28*1 = 784, length is 28, width is 28, channel is 1

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# randomize the data, 数据独立性
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train.loc[shuffle_index], y_train.loc[shuffle_index]

# cross-validation, 交叉验证
#print(y_train[:10])
y_train_5 = (y_train == '1')
y_test_5 = (y_test == '1')


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([X.loc[35000]]))
print(y.loc[35000])

from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy'))