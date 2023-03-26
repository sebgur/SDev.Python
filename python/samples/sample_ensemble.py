from sklearn.datasets import make_moons, load_iris
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
# from samples.sample_svm import plot_dataset, plot_predictions
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# from samples.sample_dataprep import split_train_test


def split_set(x_, y_, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(x_))
    test_set_size = int(len(x_) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return x_[train_indices], y_[train_indices], x_[test_indices], y_[test_indices]


x, y = make_moons(n_samples=500, noise=0.30, random_state=42)
# print(len(x))
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)  # split_set(x, y, 0.80)
# for set in (x_train, y_train, x_test, y_test):
#     print(set)
# split in training and test sets

log_clf = LogisticRegression(solver='liblinear', random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma='auto', random_state=42)

# svm_clf.fit(x_train, y_train)
# y_pred = svm_clf.predict(x_test)
# print(accuracy_score(y_test, y_pred))
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
                              voting='hard')
voting_clf.fit(x, y)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# Bagging and Pasting
def plot_decision_boundary(clf_, x_, y_, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    x_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred_ = clf_.predict(x_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred_, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contour(x1, x2, y_pred_, cmap=custom_cmap2, alpha=0.8)
    plt.plot(x_[:, 0][y_ == 0], x_[:, 1][y_ == 0], "yo", alpha=alpha)
    plt.plot(x_[:, 0][y_ == 1], x_[:, 1][y_ == 1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)


bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1)
bag_clf.fit(x_train, y_train)
y_pred = bag_clf.predict(x_test)
print(accuracy_score(y_test, y_pred))

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(x_train, y_train)
y_pred_tree = tree_clf.predict(x_test)
print(accuracy_score(y_test, y_pred_tree))

# plt.figure(figsize=(11, 4))
# plt.subplot(121)
# plot_decision_boundary(tree_clf, x, y)
# plt.title("Decision Tree", fontsize=14)
# plt.subplot(122)
# plot_decision_boundary(bag_clf, x, y)
# plt.title("Decision Trees with Bagging", fontsize=14)
# plt.show()

# Out-of-Bag Evaluation
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, bootstrap=True, n_jobs=-1, oob_score=True)
bag_clf.fit(x_train, y_train)
print(bag_clf.oob_score_)
y_pred = bag_clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
# print(bag_clf.oob_decision_function_)

# ## Note: Features can be sampled as well using two hyperparameters: max_features and bootstrap_features

# #### Random Forests ####
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(x_train, y_train)
# y_pred_rf = rnd_clf.predict(x_test)

# The above random forest is more or less equivalent to the following tree bag
bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter='random', max_leaf_nodes=16), n_estimators=500,
                            max_samples=1.0, bootstrap=True, n_jobs=-1)
bag_clf.fit(x_train, y_train)

# plt.figure(figsize=(11, 4))
# plt.subplot(121)
# plot_decision_boundary(bag_clf, x, y)
# plt.title("Decision Tree with Bagging", fontsize=14)
# plt.subplot(122)
# plot_decision_boundary(rnd_clf, x, y)
# plt.title("Random Forest", fontsize=14)
# plt.show()

# Extremely Randomized Trees (Random Forest with Extra-Trees)
extra_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
extra_clf.fit(x_train, y_train)

# plt.figure(figsize=(11, 4))
# plt.subplot(121)
# plot_decision_boundary(extra_clf, x, y)
# plt.title("Random Forest Extra-Trees", fontsize=14)
# plt.subplot(122)
# plot_decision_boundary(rnd_clf, x, y)
# plt.title("Random Forest", fontsize=14)
# plt.show()

# Feature Importance
# Train a random forest classifier. Average depth where the feature appears gives an estimate of its
# importance.
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris['data'], iris['target'])
for name, score in zip(iris['feature_names'], rnd_clf.feature_importances_):
    print(name, score)

# Boosting with AdaBoost
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200, algorithm='SAMME.R',
                             learning_rate=0.5)
ada_clf.fit(x_train, y_train)

# plt.figure(figsize=(11, 4))
# plt.subplot(121)
# plot_decision_boundary(extra_clf, x, y)
# plt.title("Random Forest Extra-Trees", fontsize=14)
# plt.subplot(122)
# plot_decision_boundary(ada_clf, x, y)
# plt.title("AdaBoost", fontsize=14)
# plt.show()

#  ## Gradient Boosting
m = 1000
length = 3
rng = np.random.RandomState(42)
x = -length + 2 * length * rng.rand(m, 1)
y = -0.0 * x**3 + 2 * x**2 + 1.5 * x + 1 + 2.1 * rng.randn(m, 1)
plot_axis = np.linspace(-length, length, num=200).reshape(-1, 1)

# with successive trees
tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(x, y)
y_tree_reg1 = tree_reg1.predict(plot_axis)

tree_reg2 = DecisionTreeRegressor(max_depth=2)
y2 = y - tree_reg1.predict(x).reshape(-1, 1)
tree_reg2.fit(x, y2)
y_tree_reg2 = sum(tree.predict(plot_axis) for tree in (tree_reg1, tree_reg2))

tree_reg3 = DecisionTreeRegressor(max_depth=2)
y3 = y2 - tree_reg2.predict(x).reshape(-1, 1)
tree_reg3.fit(x, y3)
y_tree_reg3 = sum(tree.predict(plot_axis) for tree in (tree_reg1, tree_reg2, tree_reg3))

# with GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(x, y)
y_gbrt = gbrt.predict(plot_axis)

plt.scatter(x, y, s=2, color='black', alpha=0.50, label='Sample')
plt.plot(plot_axis, y_tree_reg1, color='blue', label="Tree1")
plt.plot(plot_axis, y_tree_reg2, color='red', label="Tree2")
plt.plot(plot_axis, y_tree_reg3, color='green', label="Tree3")
plt.plot(plot_axis, y_gbrt, color='orange', label="GBRT")
plt.title("Compare regressions")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.show()


