from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
import numpy as np
import matplotlib.pyplot as plt

export_path = r'D:\sandbox\Outputs\iris_tree.dot'
iris = datasets.load_iris()
# keys = list(iris.keys())
# print(keys)

# ## Decision Tree ##
x = iris['data'][:, 2:]  # petal length and width
y = iris['target']

# criterion='gini' as default, or 'entropy'
tree_clf = DecisionTreeClassifier(max_depth=2, criterion='entropy')
tree_clf.fit(x, y)
# export_graphviz(tree_clf, out_file=export_path, feature_names=iris.feature_names[2:],
#                 class_names=iris.target_names, rounded=True, filled=True)

# Gini impurity
print(1-(1/46)**2-(45/46)**2)

# Probabilities
print(tree_clf.predict_proba([[5, 1.5]]))
print(tree_clf.predict([[5, 1.5]]))
print(49/54)

# Tree regression
m = 1000
length = 3
rng = np.random.RandomState(42)
x = -length + 2 * length * rng.rand(m, 1)
y = -1.5 * x**3 + 2 * x**2 + 3 * x + 1 + 6.2 * rng.randn(m, 1)
plot_axis = np.linspace(-length, length, num=100).reshape(-1, 1)

tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(x, y)
y_tree_reg2 = tree_reg2.predict(plot_axis)

tree_reg3 = DecisionTreeRegressor(max_depth=3)
tree_reg3.fit(x, y)
y_tree_reg3 = tree_reg3.predict(plot_axis)

tree_reg10 = DecisionTreeRegressor(max_depth=10)
tree_reg10.fit(x, y)
y_tree_reg10 = tree_reg10.predict(plot_axis)

tree_regNo = DecisionTreeRegressor()
tree_regNo.fit(x, y)
y_tree_regNo = tree_regNo.predict(plot_axis)

plt.scatter(x, y, s=2, color='black', alpha=0.50, label='Sample')
plt.plot(plot_axis, y_tree_reg2, color='blue', label="Tree2")
plt.plot(plot_axis, y_tree_reg3, color='red', label="Tree3")
plt.plot(plot_axis, y_tree_reg10, color='green', label="Tree10")
plt.plot(plot_axis, y_tree_regNo, color='orange', label="TreeNo")
# plt.plot(plot_axis, y_poly, color='red', label="Poly")
plt.title("Compare regressions")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.show()
