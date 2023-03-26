import numpy as np
from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris["data"][:, (2, 3)]  # petal length, petal width at coordinates (2, 3) of each row of data
y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica
# print(np.shape(x))
# print(np.shape(y))
# Linear classification
def plot_svc_decision_boundary(svm_clf, scaler_, xmin, xmax):
    # At the decision boundary, w0*x0 + w1*x1 + b = 0 in scaled coordinates with
    # w = svm_clf.coef_[0]
    # b = svm_clf.intercept_[0]
    # Convert to unscaled
    w = svm_clf.coef_[0] / scaler_.scale_
    b = svm_clf.decision_function([-scaler_.mean_ / scaler_.scale_])

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]

    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    # svs = svm_clf.support_vectors_
    # plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)


scaler = StandardScaler()
svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)

scaled_svm_clf1 = Pipeline([("scaler", scaler), ("linear_svc", svm_clf1)])
scaled_svm_clf2 = Pipeline([("scaler", scaler), ("linear_svc", svm_clf2)])

scaled_svm_clf1.fit(x, y)
scaled_svm_clf2.fit(x, y)

# plt.figure(figsize=(12, 3.2))
# plt.subplot(121)
# plt.plot(x[:, 0][y == 1], x[:, 1][y == 1], "g^", label="Iris-Virginica")
# plt.plot(x[:, 0][y == 0], x[:, 1][y == 0], "bs", label="Iris-Versicolor")
# plot_svc_decision_boundary(svm_clf1, scaler, 4, 6)
# plt.xlabel("Petal length", fontsize=14)
# plt.ylabel("Petal width", fontsize=14)
# plt.legend(loc="upper left", fontsize=14)
# plt.title("$C = {}$".format(svm_clf1.C), fontsize=16)
# plt.axis([4, 6, 0.8, 2.8])
#
# plt.subplot(122)
# plt.plot(x[:, 0][y == 1], x[:, 1][y == 1], "g^")
# plt.plot(x[:, 0][y == 0], x[:, 1][y == 0], "bs")
# plot_svc_decision_boundary(svm_clf2, scaler, 4, 6)
# plt.xlabel("Petal length", fontsize=14)
# plt.title("$C = {}$".format(svm_clf2.C), fontsize=16)
# plt.axis([4, 6, 0.8, 2.8])
# plt.show()

# Nonlinear SVM Classification
def plot_dataset(x_, y_, axes):
    plt.plot(x_[:, 0][y_ == 0], x_[:, 1][y_ == 0], "bs")
    plt.plot(x_[:, 0][y_ == 1], x_[:, 1][y_ == 1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)


x, y = make_moons(n_samples=100, noise=0.15, random_state=42)
# plot_dataset(x, y, [-1.5, 2.5, -1, 1.5])
# plt.show()


def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 200)
    x1s = np.linspace(axes[2], axes[3], 200)
    x0, x1 = np.meshgrid(x0s, x1s)
    x_ = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(x_).reshape(x0.shape)
    y_decision = clf.decision_function(x_).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


# Linear with polynomial
poly_svm_clf = Pipeline([("poly_features", PolynomialFeatures(degree=3)),
                         ("scaler", StandardScaler()),
                         ("svm_clf", LinearSVC(C=10, loss='hinge', random_state=42))])
poly_svm_clf.fit(x, y)
# plot_predictions(poly_svm_clf, [-1.5, 2.5, -1, 1.5])
# plot_dataset(x, y, [-1.5, 2.5, -1, 1.5])
# plt.show()

# Non-linear with kernels
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
kernel_svm_clf = SVC(kernel='poly', degree=3, coef0=1, C=5)
kernel_svm_clf.fit(x_scaled, y)
poly_kernel_svm_clf = Pipeline([("scaler", StandardScaler()),
                                ("svm_clf", SVC(kernel='poly', degree=3, coef0=1, C=5))])
poly_kernel_svm_clf.fit(x, y)
poly100_kernel_svm_clf = Pipeline([("scaler", StandardScaler()),
                                   ("svm_clf", SVC(kernel='poly', degree=10, coef0=100, C=5))])
poly100_kernel_svm_clf.fit(x, y)

# Plot
plt.figure(figsize=(11, 4))

plt.subplot(121)
plot_predictions(poly_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(x, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=3, r=1, C=5$", fontsize=18)

plt.subplot(122)
plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(x, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=10, r=100, C=5$", fontsize=18)

plt.show()

# ## SVM Regression ##
m = 1000
length = 3
rng = np.random.RandomState(42)
s = -length + 2 * length * rng.rand(m, 1)
f = -1.5 * s**3 + 2 * s**2 + 3 * s + 1 + 6.2 * rng.randn(m, 1)
plot_axis = np.linspace(-length, length, num=100).reshape(-1, 1)

# Linear
svm_reg = LinearSVR(epsilon=1.5)
pipe_svm_reg = Pipeline([("scaler", StandardScaler()),
                         ("svm_reg", svm_reg)])
pipe_svm_reg.fit(s, f)
svm_reg_pred = pipe_svm_reg.predict(plot_axis)

# Polynomial
poly_svm_reg = LinearSVR(epsilon=1.5)
pipe_poly_svm_reg = Pipeline([("poly_features", PolynomialFeatures(3)),
                              ("scaler", StandardScaler()),
                              ("svm_reg", poly_svm_reg)])
pipe_poly_svm_reg.fit(s, f)
poly_svm_reg_pred = pipe_poly_svm_reg.predict(plot_axis)

# RBF Kernel
kernel_svm_reg = SVR(kernel='rbf', gamma=100, C=0.5)
pipe_kernel_svm_reg = Pipeline([("scaler", StandardScaler()),
                                ("svm_reg", kernel_svm_reg)])
pipe_kernel_svm_reg.fit(s, f)
kernel_svm_reg_pred = pipe_kernel_svm_reg.predict(plot_axis)

# Plot
# plt.scatter(s, f, s=2, color="black", label="Sample")
# plt.plot(plot_axis, svm_reg_pred, color='blue', label='Linear')
# plt.plot(plot_axis, poly_svm_reg_pred, color='red', label='Poly')
# plt.plot(plot_axis, kernel_svm_reg_pred, color='green', label='RBF')
# plt.legend(loc='best')
# plt.show()
