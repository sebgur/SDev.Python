from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
keys = list(iris.keys())
print(keys)

# Logistic Regression
x = iris['data'][:, 3:]  # petal width
y = (iris['target'] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0
log_reg = LogisticRegression(solver='lbfgs')
log_reg.fit(x, y)
x_new = np.linspace(0, 3, 100).reshape(-1, 1)
y_proba = log_reg.predict_proba(x_new)[:, 1]
y_bin = log_reg.predict(x_new)
y_result = dict(zip(y_proba, y_bin))
y_res2 = [[y_proba[i], y_bin[i]] for i in range(len(y_proba))]
# plt.plot(x_new, y_proba, "g-", label="Iris-Virginica")
# plt.plot(y_proba, y_bin, "g-", label="Iris-Virginica")
# plt.show()

# Softmax Regression (aka Multinomial Logistic Regression)
x = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris['target']
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)  # if no multi_class, uses OvA
softmax_reg.fit(x, y)
test_points = [[5, 2]]
res = softmax_reg.predict(test_points)
prob = softmax_reg.predict_proba(test_points)
print(res)
print(prob)
