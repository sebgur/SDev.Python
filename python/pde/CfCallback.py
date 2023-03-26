import keras
from tools.utils import rmse
from sklearn.preprocessing import MinMaxScaler
# import time


class CfCallback(keras.callbacks.Callback):
    def __init__(self, x_test, y_ref_test):
        self.x_test = x_test
        self.y_ref_test = y_ref_test
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.epochs = []
        self.accuracies = []
        # self.offset = 0

    def on_train_begin(self, logs=None):
        self.epochs.clear()
        self.accuracies.clear()

    # def on_train_end(self, logs=None):
    #     keys = list(logs.keys())
    #     self.offset = len(self.epochs)

    # def on_epoch_begin(self, epoch, logs=None):
    #     keys = list(logs.keys())
    #     print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        # time.sleep(1.5)
        x_scaled = self.x_scaler.transform(self.x_test)
        y_scaled = self.model.predict(x_scaled, verbose=0)
        y_mod_test = self.y_scaler.inverse_transform(y_scaled)
        rmse_ = rmse(self.y_ref_test, y_mod_test)
        self.epochs.append(epoch)
        # self.epochs.append(self.offset + epoch)
        self.accuracies.append(rmse_)
        print(" ")
        # print(logs['loss'])
        print("Estimated rmse: {a:,.0f}".format(a=rmse_))
        # print("Estimated rmse: {a:,.2f}".format(a=rmse_))

    def set_scalers(self, x_scaler, y_scaler):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def convergence(self):
        return self.epochs, self.accuracies
