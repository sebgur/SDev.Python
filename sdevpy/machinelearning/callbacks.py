""" Custom callbacks for Keras training """
# import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler
# from sdevpy.maths.metrics import rmse


class SDevPyCallback(keras.callbacks.Callback):
    """ SDevPy's base class for callbacks. Compared to Keras's base, it also keeps track of 
        the learning rate and displays it at each period """
    def __init__(self, epoch_sampling=1, optimizer=None):
        keras.callbacks.Callback.__init__(self)
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.optimizer = optimizer
        self.epoch_sampling = epoch_sampling
        self.total_epochs = 0
        self.batch_size = 0
        self.set_size = 0
        self.shuffle = True
        self.epochs = []
        self.losses = []
        self.learning_rates = []
        self.sampled_epochs = []

    def on_train_begin(self, logs=None):
        """ Display settings, reset training variables """
        print("<><><><><><><><> TRAINING START <><><><><><><><>")
        self.epochs.clear()
        self.losses.clear()
        self.learning_rates.clear()
        self.sampled_epochs.clear()
        print(f"Epochs: {self.total_epochs:,}")
        print(f"Batch size: {self.batch_size:,}")
        print(f"Shuffle: {'True' if self.shuffle else 'False'}")
        print(f"Training set size: {self.set_size:,}", end="")

    def on_train_end(self, logs=None):
        """ Display results """
        print("\n<><><><><><><><> TRAINING END <><><><><><><><>")

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.epoch_sampling == 0:
            print("\n<><><><><><><><><><><><><><><><>")
            print(f"Epoch {epoch:,}/{self.total_epochs:,}")

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        loss = logs['loss']
        scale = self.y_scaler.scale_[0]
        loss = loss * scale
        self.losses.append(loss)
        if self.optimizer is not None:
            lr = self.optimizer.learning_rate.numpy()
            self.learning_rates.append(lr)

        if epoch % self.epoch_sampling == 0:
            self.sampled_epochs.append(epoch)
            print(f"Loss: {loss:,.2f}", end="")
            if self.optimizer is not None:
                print(f", LR: {lr:.6f}", end="")

            # print(list(logs.keys()))

    def set_scalers(self, x_scaler, y_scaler):
        """ Set scalers """
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    # def convergence(self):
    #     """ Retrieve sampled epochs, losses and learning rates """
    #     return self.epochs, self.losses, self.learning_rates, self.sampled_epochs


class RefCallback(SDevPyCallback):
    """ Callback class that compares model predictions vs reference periodically """
    def __init__(self, x_test, y_test, loss_function, epoch_sampling=1,
                 optimizer=None):
        SDevPyCallback.__init__(self, epoch_sampling, optimizer)
        self.x_test = x_test
        self.y_test = y_test
        self.loss_function = loss_function
        self.test_losses = []

    def on_train_begin(self, logs=None):
        SDevPyCallback.on_train_begin(self, logs)
        self.test_losses.clear()

    # def on_train_end(self, logs=None):
    #     SDevPyCallback.on_train_end(self, logs)
    #     test_loss = self.estimate_loss(self.x_test, self.y_test)
    #     print(f", Test loss: {test_loss:,.2f}")

    def on_epoch_end(self, epoch, logs=None):
        SDevPyCallback.on_epoch_end(self, epoch, logs)
        if epoch % self.epoch_sampling == 0:
            test_loss = self.estimate_loss(self.x_test, self.y_test)
            self.test_losses.append(test_loss)
            print(f", Test loss: {test_loss:,.2f}", end="")

    def estimate_loss(self, x_est, y_est):
        """ Estimate loss on test set """
        x_scaled = self.x_scaler.transform(x_est)
        y_scaled = self.model.predict(x_scaled, verbose=0)
        y_pred = self.y_scaler.inverse_transform(y_scaled)
        est_loss = self.loss_function(y_est, y_pred)
        # est_loss = rmse(y_est, y_pred) * 10000
        return est_loss

    def test_set_loss(self):
        """ Retrieve loss on test set """
        return self.test_losses
