import pandas as pd
import numpy as np
import matplotlib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score

from keras.layers import Dense, Input, Dropout, Embedding, LSTM
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam, Nadam
from keras.callbacks import TensorBoard
from keras import regularizers

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)  # DEBUG, INFO, WARN, ERROR, FATAL

# Visualize results
def print_report(y_true_, y_pred_):
    r = precision_recall_fscore_support(y_true_, y_pred_)
    print('Accuracy: {:.2%}'.format(accuracy_score(y_true_, y_pred_)))
    print('AUC: {:.2%}'.format(roc_auc_score(y_true_, y_pred_)))
    print('C0| Precision: {:.2%} Recall: {:.2%} F1-score: {:.2%} Support: {:,}'
          .format(r[0][0], r[1][0], r[2][0], r[3][0]))
    print('C1| Precision: {:.2%} Recall: {:.2%} F1-score: {:.2%} Support: {:,}'
          .format(r[0][1], r[1][1], r[2][1], r[3][1]))


def print_autoencoder_report(x_test_, y_test_, y_pred_, threshold):
    y_dist = np.linalg.norm(x_test_ - y_pred_, axis=-1)
    z = zip(y_dist >= threshold, y_dist)
    y_label = []
    error = []
    for idx, (is_anomaly, y_dist) in enumerate(z):
        if is_anomaly:
            y_label.append(1)
        else:
            y_label.append(0)
        error.append(y_dist)

    print('Threshold: {:2f}'.format(threshold))
    print_report(y_test_, y_label)


# Read the input data
path = r"C:\Users\sebgu\Desktop\Datasets"
file = r"C:\Users\sebgu\Desktop\Datasets\creditcard.csv"
df = pd.read_csv(filepath_or_buffer=file, header=0, sep=',')
# print(df.shape[0])
# print(df.head())

# Create training and testing datasets
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df0 = df.query('Class == 0').sample(20000)
df1 = df.query('Class == 1').sample(400)
df = pd.concat([df0, df1])

x_train, x_test, y_train, y_test = train_test_split(df.drop(labels=['Time', 'Class'], axis=1), df['Class'],
                                                    test_size=0.2, random_state=42)
# print(x_train.shape, 'training sample')
# print(x_test.shape, 'test sample')
# print(y_train.shape, 'training sample y')
# print(y_test.shape, 'test sample y')

# Set up network
encoding_dim = 12
input_dim = x_train.shape[1]
input_array = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_array)
decoded = Dense(input_dim, activation='softmax')(encoded)

autoencoder = Model(input_array, decoded)
print(autoencoder.summary())
autoencoder.compile(optimizer=RMSprop(), loss='mean_squared_error', metrics=['mae', 'accuracy'])

# Train
batch_size = 32
epochs = 20
history = autoencoder.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, verbose=0, shuffle=True,
                          validation_data=(x_test, x_test), callbacks=[TensorBoard(log_dir=path)])

# print("Evaluate")
# score = autoencoder.evaluate(x_test, x_test, verbose=1)
# print('Test loss: ', score[0])
# print('Test accuracy: ', score[1])

# Test predictions
y_pred = autoencoder.predict(x_test)
print()
print_autoencoder_report(x_test, y_test, y_pred, 2.0)
print()
print_autoencoder_report(x_test, y_test, y_pred, 5.0)
print()
print_autoencoder_report(x_test, y_test, y_pred, 7.0)
print()
print_autoencoder_report(x_test, y_test, y_pred, 10.0)
print()
print_autoencoder_report(x_test, y_test, y_pred, 15.0)

# # Sparse encoders with regularization
# input_array = Input(shape=(input_dim,))
# encoded = Dense(encoding_dim, activation='relu',
#                 activity_regularizer=regularizers.l1(10e-5))(input_array)
# decoded = Dense(input_dim, activation='softmax')(encoded)
#
# autoencoder = Model(input_array, decoded)
# print(autoencoder.summary())
# autoencoder.compile(optimizer=RMSprop(), loss='mean_squared_error', metrics=['mae', 'accuracy'])
#
# # Train
# batch_size = 32
# epochs = 20
# history = autoencoder.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, verbose=0, shuffle=True,
#                           validation_data=(x_test, x_test), callbacks=[TensorBoard(log_dir=path)])
#
# # Test predictions
# y_pred = autoencoder.predict(x_test)
# print()
# print_autoencoder_report(x_test, y_test, y_pred, 2.0)
# print()
# print_autoencoder_report(x_test, y_test, y_pred, 5.0)
# print()
# print_autoencoder_report(x_test, y_test, y_pred, 7.0)
# print()
# print_autoencoder_report(x_test, y_test, y_pred, 10.0)
# print()
# print_autoencoder_report(x_test, y_test, y_pred, 15.0)

# Dense encoder
encoding_dim = 16
input_array = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_array)
encoded = Dense(8, activation='relu')(encoded)
encoded = Dense(4, activation='relu')(encoded)

decoded = Dense(8, activation='relu')(encoded)
decoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(input_dim, activation='softmax')(encoded)

autoencoder = Model(input_array, decoded)
print(autoencoder.summary())
autoencoder.compile(optimizer=RMSprop(), loss='mean_squared_error', metrics=['mae', 'accuracy'])

# Train
batch_size = 32
epochs = 20
history = autoencoder.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, verbose=0, shuffle=True,
                          validation_data=(x_test, x_test), callbacks=[TensorBoard(log_dir=path)])

# Test predictions
y_pred = autoencoder.predict(x_test)
print()
print_autoencoder_report(x_test, y_test, y_pred, 2.0)
print()
print_autoencoder_report(x_test, y_test, y_pred, 5.0)
print()
print_autoencoder_report(x_test, y_test, y_pred, 7.0)
print()
print_autoencoder_report(x_test, y_test, y_pred, 10.0)
print()
print_autoencoder_report(x_test, y_test, y_pred, 15.0)