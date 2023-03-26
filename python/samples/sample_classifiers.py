# from sklearn.datasets import fetch_mldata
# import os
import sklearn
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib
import matplotlib.pyplot as plt


# Load the dataset
def sort_by_target(mnist_):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist_.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist_.target[60000:])]))[:, 1]
    mnist_.data[:60000] = mnist_.data[reorder_train]
    mnist_.target[:60000] = mnist_.target[reorder_train]
    mnist_.data[60000:] = mnist_.data[reorder_test + 60000]
    mnist_.target[60000:] = mnist_.target[reorder_test + 60000]


custom_data_home = r"F:\Sebastien\OneDrive\Work\ML\Geron - Hands-On Machine Learning\datasets\mnist"
mnist = fetch_openml('mnist_784', version=1, cache=True, data_home=custom_data_home)
mnist.target = mnist.target.astype(np.int8)  # fetch_openml() returns targets as strings
x, y = mnist['data'], mnist['target']
# print(x.shape)
# print(y.shape)

# View image
idx = 36000
some_digit = x[idx]
some_digit_truevalue = y[idx]
print("Number for item " + str(idx) + ": " + str(some_digit_truevalue))
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
# plt.axis('off')
# plt.show()

# ###################################################################################################################
# ########################## Single class  ##########################################################################
# ###################################################################################################################
# Training and test sets
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
shuffle_index = np.random.RandomState(42).permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

# Train binary classifier
print()
print("Training classifier...")
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)
test_predict = sgd_clf.predict([some_digit])
print("Item " + str(idx) + " predicted to be 5: " + ("true" if test_predict[0] else "false"))

# Cross-validation
print()
print("Cross-validation")
scores = cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring='accuracy')
print(scores)

# Measure performance
print()
print("Performance on single fit with confusion matrix")
train_size = len(y_train_5)
y_train_pred = sgd_clf.predict(x_train)
result_diff = np.ndarray(shape=(len(y_train_pred), 1))
tp, fn, fp, tn = 0, 0, 0, 0
for i in range(len(y_train_pred)):
    if y_train_5[i]:
        tp = tp + (1 if y_train_pred[i] else 0)
        fn = fn + (1 if not y_train_pred[i] else 0)
    else:
        fp = fp + (1 if y_train_pred[i] else 0)
        tn = tn + (1 if not y_train_pred[i] else 0)
    result_diff[i] = 1 if y_train_pred[i] == y_train_5[i] else 0
# conf_matrix = [[tn, fp], [fn, tp]]
# print(conf_matrix)
conf_matrix = confusion_matrix(y_train_5, y_train_pred)
print(conf_matrix)
print('Precision: ' + str(precision_score(y_train_5, y_train_pred)) + " <><> " + str(tp / (tp + fp)))
print('Recall: ' + str(recall_score(y_train_5, y_train_pred)) + " <><> " + str(tp / (tp + fn)))

# Cross-prediction
print()
print("Cross-prediction")
y_train_cross = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3)
conf_matrix = confusion_matrix(y_train_5, y_train_cross)
print(conf_matrix)

# Decision scores and precision-recall curve
y_scores = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
# plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
# plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
# plt.xlabel("Threshold")
# plt.legend(loc="upper left")
# plt.ylim([0, 1])
# plt.show()

# Receiver Operating Characteristic (ROC) curve
# fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
# plt.plot(fpr, tpr, linewidth=2, label=None)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.axis([0, 1, 0, 1])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.show()

# ###################################################################################################################
# ########################## Multi-class  ###########################################################################
# ###################################################################################################################
# One-versus-All
print()
print("One-versus-All")
sgd_clf.fit(x_train, y_train)
new_prediction = sgd_clf.predict([some_digit])
print("Prediction for item " + str(idx) + ": " + str(new_prediction) + " vs true value: " + str(some_digit_truevalue))
print("Classes")
print(sgd_clf.classes_)
print("Scores")
print(sgd_clf.decision_function([some_digit]))

# Random forest
print()
print("Random Forest")
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(x_train, y_train)
new_prediction = forest_clf.predict([some_digit])
print("Prediction for item " + str(idx) + ": " + str(new_prediction) + " vs true value: " + str(some_digit_truevalue))
print("Probabilities")
print(forest_clf.predict_proba([some_digit]))

# Error analysis
y_train_pred = cross_val_predict(sgd_clf, x_train, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
