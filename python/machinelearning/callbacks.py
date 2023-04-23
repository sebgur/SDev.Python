""" Custom callbacks for Keras training """
import keras
from sklearn.preprocessing import MinMaxScaler
from maths.metrics import rmse


class SDevPyCallback(keras.callbacks.Callback):
    """ SDevPy's base class for callbacks. Compared to Keras's base, it also keeps track of 
        the learning rate and displays it at each period """
    def __init__(self, epoch_sampling=1):
        keras.callbacks.Callback.__init__(self)
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.epoch_sampling = epoch_sampling
        self.epochs = []
        self.losses = []
        self.learning_rates = []

    def on_train_begin(self, logs=None):
        """ Reset training variables if needed """
        self.epochs.clear()
        self.losses.clear()
        self.learning_rates.clear()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_sampling == 0:
            print(f"Epoch {epoch:,}/{self.total_epochs:,}")
            self.epochs.append(epoch)
            loss = logs['loss']
            self.losses.append(loss)
            print(f"Loss: {loss * 10000.0:,.2f}")
            # print(list(logs.keys()))
            # for k, v in logs.items():
            #     print(k, end=": ")
            #     print(v)
            # print(f"Sample epoch: {rmse_:,.0f}")


    def set_scalers(self, x_scaler, y_scaler):
        """ Set scalers """
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def set_total_epochs(self, total_epochs):
        """ Set number of epochs in current training phase """
        self.total_epochs = total_epochs

    def convergence(self):
        """ Retrieve sampled epochs, losses and learning rates """
        return self.epochs, self.losses, self.learning_rates


class RefCallback(SDevPyCallback):
    """ Callback class that compares model predictions vs reference periodically """
    def __init__(self, x_test, y_ref_test):
        SDevPyCallback.__init__(self)
        self.x_test = x_test
        self.y_ref_test = y_ref_test
        # self.x_scaler = MinMaxScaler()
        # self.y_scaler = MinMaxScaler()
        # self.epochs = []
        self.accuracies = []
        # self.offset = 0

    # def on_train_begin(self, logs=None):
    #     self.epochs.clear()
    #     self.accuracies.clear()

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
        print(f"Estimated rmse: {rmse_:,.0f}")

    # def set_scalers(self, x_scaler, y_scaler):
    #     """ Set scalers """
    #     self.x_scaler = x_scaler
    #     self.y_scaler = y_scaler

    # def convergence(self):
    #     """ Retrieve convergence parameters """
    #     return self.epochs, self.accuracies
