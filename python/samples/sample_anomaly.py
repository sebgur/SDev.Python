import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans

from pyod.models.cblof import CBLOF
from pyod.models.knn import KNN
from pyod.models.copod import COPOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.sod import SOD
from pyod.models.sos import SOS
from pyod.models.mad import MAD

x, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=2, weights=[0.98, ], class_sep=0.5,
                           scale=1.0, shuffle=True, flip_y=0, random_state=0)

# Plot
# plt.scatter(x[:, 0][y == 0], x[:, 1][y == 0], color='blue', alpha=0.1, label='0s')
# plt.scatter(x[:, 0][y == 1], x[:, 1][y == 1], color='red', label='1s')
# plt.legend(loc='best')
# plt.show()

# Naive percentile based detection
print('Percentiles')
class PercentileDetection:
    def __init__(self, percentile=0.9):
        self.percentile = percentile
        self.thresholds = None

    def fit(self, x_, y_=None):
        self.thresholds = [pd.Series(x_[:, i]).quantile(self.percentile) for i in range(x.shape[1])]

    def predict(self, x_, y_=None):
        return (x_ > self.thresholds).max(axis=1)

    def fit_predict(self, x_, y_=None):
        self.fit(x_)
        return self.predict(x_)


prc1 = PercentileDetection(0.98)
y_prc1 = prc1.fit_predict(x)
print('Precision: {0:.2%}, Recall: {1:.2%}'.format(precision_score(y, y_prc1), recall_score(y, y_prc1)))

# Elliptic Envelope
print("Elliptic Envelope")
ee = EllipticEnvelope(random_state=42)
y_ee = ee.fit_predict(x) == -1
print('Precision: {0:.2%}, Recall: {1:.2%}'.format(precision_score(y, y_ee), recall_score(y, y_ee)))

# Local Outlier Factor (LOF)
print("LOF")
lof = LocalOutlierFactor(n_neighbors=50, contamination='auto')
y_lof = lof.fit_predict(x) == -1
print('Precision: {0:.2%}, Recall: {1:.2%}'.format(precision_score(y, y_lof), recall_score(y, y_lof)))

# Novelty detection with Local Outlier Factor (LOF)
print("Novelty LOF")
nlof = LocalOutlierFactor(n_neighbors=50, contamination='auto', novelty=True)
nlof.fit(x[y == 0])
y_nlof = nlof.predict(x) == -1
print('Precision: {0:.2%}, Recall: {1:.2%}'.format(precision_score(y, y_nlof), recall_score(y, y_nlof)))

# Isolation Forest
print("Isolation Forest")
isof = IsolationForest(n_estimators=200, contamination='auto', behaviour='new', n_jobs=-1, random_state=10)
y_isof = isof.fit_predict(x) == -1
print('Precision: {0:.2%}, Recall: {1:.2%}'.format(precision_score(y, y_isof), recall_score(y, y_isof)))

# Cluster-Based LOF (CBLOF, pyod)
print("Cluster-Based LOF (pyod)")
cblof = CBLOF()
y_cblof = cblof.fit_predict(x)
print('Precision: {0:.2%}, Recall: {1:.2%}'.format(precision_score(y, y_cblof), recall_score(y, y_cblof)))

# Copula-Based LOF (CBLOF, pyod)
print("COPOD (pyod)")
copod = COPOD()
y_copod = copod.fit_predict(x)
print('Precision: {0:.2%}, Recall: {1:.2%}'.format(precision_score(y, y_copod), recall_score(y, y_copod)))

# K-neighbours (KNN, pyod)
print("KNN (pyod)")
knn = KNN()
y_knn = knn.fit_predict(x)
print('Precision: {0:.2%}, Recall: {1:.2%}'.format(precision_score(y, y_knn), recall_score(y, y_knn)))

# SOD (pyod)
print("SOD (pyod)")
sod = SOD()
y_sod = sod.fit_predict(x)
print('Precision: {0:.2%}, Recall: {1:.2%}'.format(precision_score(y, y_sod), recall_score(y, y_sod)))

# SOS (pyod)
print("SOS (pyod)")
sos = SOS()
y_sos = sos.fit_predict(x)
print('Precision: {0:.2%}, Recall: {1:.2%}'.format(precision_score(y, y_sos), recall_score(y, y_sos)))

# K-means
# Does not seem to run if n_clusters is larger than n_features. But at n_clusters = 2, it just
# splits into 2 equal groups. Not sure how to properly used it for anomaly detection.
# print("K-means")
# kmeans = KMeans(n_clusters=2)
# y_kmeans = kmeans.fit_predict(x)
# print(sum)
# print('Precision: {0:.2%}, Recall: {1:.2%}'.format(precision_score(y, y_kmeans), recall_score(y, y_kmeans)))



# # Auto-encoder (pyod)
# print("Auto-encoder (pyod)")
# auenc = AutoEncoder()
# y_auenc = auenc.fit_predict(x)
# print('Precision: {0:.2%}, Recall: {1:.2%}'.format(precision_score(y, y_auenc), recall_score(y, y_auenc)))

# ## From 'Beginning Anomaly Detection Using Python", by Alla
file = r"C:\Users\sebgu\Desktop\Datasets\kddcup.data.corrected"
# print()
# print("KDD dataset for Isolation Forest")
# columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
#            "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
#            "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
#            "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
#            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
#            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
#            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
#            "dst_host_rerror_rate", "dst_host_srv_error_rate", "label"]
# df = pd.read_csv(file, sep=",", names=columns, index_col=None)
# print(df.shape)
# df = df[df["service"] == "http"]
# df = df.drop("service", axis=1)
# columns.remove("service")
# print(df.shape)
# # Check types of attacks
# print(df["label"].value_counts())
# # Encode categorical features
# for col in df.columns:
#     if df[col].dtype == "object":
#         encoded = LabelEncoder()
#         encoded.fit(df[col])
#         df[col] = encoded.transform(df[col])
#
# for f in range(0, 3):
#     df = df.iloc[np.random.permutation(len(df))]
#
# df2 = df[:500000]
# labels = df2["label"]
# df_validate = df[500000:]
# x_train, x_test, y_train, y_test = train_test_split(df2, labels, test_size=0.2, random_state=42)
# x_val, y_val = df_validate, df_validate["label"]
# print(x_train.shape)
# print(x_test.shape)
# print(x_val.shape)

# isolation_forest = IsolationForest(n_estimators=100, max_samples=256, contamination=0.1, random_state=42)
# isolation_forest.fit(x_train)
# anomaly_scores = isolation_forest.decision_function(x_val)
# plt.figure(figsize=(15, 10))
# plt.hist(anomaly_scores, bins=100)
# plt.xlabel('Average Path Lengths', fontsize=14)
# plt.ylabel('Number of Data Points', fontsize=14)
# plt.show()

# # Metrics
# anomalies = anomaly_scores > -0.19
# matches = y_val == list(encoded.classes_).index("normal.")
# auc = roc_auc_score(anomalies, matches)
# print("AUC on validation set: {:.2%}".format(auc))
#
# anomaly_scores_test = isolation_forest.decision_function(x_test)
# anomalies_test = anomaly_scores_test > -0.19
# matches = y_test == list(encoded.classes_).index("normal.")
# auc = roc_auc_score(anomalies_test, matches)
# print("AUC on test set: {:.2%}".format(auc))

# # One-Class SVM (OC-SVM)
# print()
# print("KDD dataset for OC-SVM")
# columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
#            "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
#            "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
#            "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
#            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
#            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
#            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
#            "dst_host_rerror_rate", "dst_host_srv_error_rate", "label"]
#
# df = pd.read_csv(file, sep=",", names=columns, index_col=None)
# print(df.shape)
# df = df[df["service"] == "http"]
# df = df.drop("service", axis=1)
# columns.remove("service")
# print(df.shape)
#
# novelties = df[df["label"] != "normal."]  # 4045, all anomalies
# novelties_normal = df[150000:154045]  # 4045, all normal
# novelties = pd.concat([novelties, novelties_normal])  # 8090, half anomalies half normal
# normal = df[df["label"] == "normal."]  # 619046, all normal
# print(novelties.shape)
# print(normal.shape)
#
# # Encode categorical values
# for col in normal.columns:
#     if normal[col].dtype == "object":
#         encoded = LabelEncoder()
#         encoded.fit(normal[col])
#         normal[col] = encoded.transform(normal[col])
#
# for col in novelties.columns:
#     if novelties[col].dtype == "object":
#         encoded2 = LabelEncoder()
#         encoded2.fit(novelties[col])
#         novelties[col] = encoded2.transform(novelties[col])
#
# # Shuffle entries in normal dataset
# for f in range(0, 10):
#     normal = normal.iloc[np.random.permutation(len(normal))]
#
# df2 = pd.concat([normal[:100000], normal[200000:250000]])
# df_validate = normal[100000:150000]
# x_train, x_test = train_test_split(df2, test_size = 0.2, random_state=42)
# x_val = df_validate
# print(x_train.shape)
# print(x_test.shape)
# print(x_val.shape)
#
# ocsvm = OneClassSVM(kernel='rbf', gamma=0.00005, random_state=42, nu=0.1)
# ocsvm.fit(x_train)
#
# preds = ocsvm.predict(x_test)
# score = 0
# for f in range(0, x_test.shape[0]):
#     if preds[f] == 1:
#         score += 1
#
# accuracy = score / x_test.shape[0]
# print("Accuracy: {:.2%}".format(accuracy))
#
# preds = ocsvm.predict(x_val)
# score = 0
# for f in range(0, x_val.shape[0]):
#     if preds[f] == 1:
#         score += 1
#
# accuracy = score / x_val.shape[0]
# print("Accuracy: {:.2%}".format(accuracy))
#
# preds = ocsvm.predict(novelties)
# matches = novelties['label'] == 4
# auc = roc_auc_score(preds, matches)
# print("AUC: {:.2%}".format(auc))
#
# plt.figure(figsize=(10, 5))
# plt.hist(preds, bins=[-1.5, -0.5] + [0.5, 1.5], align='mid')
# plt.xticks([-1, 1])
# plt.show()
