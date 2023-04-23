""" Wrapper class for machine learning models, including scalers, and simplifying
    evaluation, history tracking, exporting to/importing from files, etc. """
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import tensorflow as tf
import joblib
from machinelearning.callbacks import SDevPyCallback

# ToDo: do more in the call back. Currently it is too specific, so different call-backs wouldn't
# work. Instead we should do everything in the call-back, and then retrieve information from it
# in the end, knowing its special type.

class LearningModel:
    """ Wrapper class for machine learning models, including scalers, and simplifying
        evaluation, history tracking, exporting to/importing from files, etc. """
    def __init__(self, model,
                 x_scaler=StandardScaler(copy=True),
                 y_scaler=StandardScaler(copy=True)):
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.is_scaled = False
        self.epochs = []
        self.losses = []
        self.accuracies = []
        self.offset = 0

    def train(self, x_set, y_set, epochs, batch_size, callback=None,
              verbose=0, shuffle=True):
        """ Scale on first call, then train """
        if not self.is_scaled:
            self.x_scaler.fit(x_set)
            self.y_scaler.fit(y_set)
            self.is_scaled = True
            callbacks = []
            if callback is not None:
                callback.set_scalers(self.x_scaler, self.y_scaler)
                callback.set_total_epochs(epochs)
                callbacks = [callback]

        x_scaled = self.x_scaler.transform(x_set)
        y_scaled = self.y_scaler.transform(y_set)

        history = self.model.fit(x_scaled, y_scaled, epochs=epochs, batch_size=batch_size,
                                 shuffle=shuffle, verbose=verbose, callbacks=callbacks)

        # epochs, accuracies = call_back.convergence()

        self.losses.extend(history.history['loss'])
        # self.accuracies.extend(accuracies)

        # num_epochs = len(epochs)
        # for i in range(num_epochs):
        #     self.epochs.append(self.offset + epochs[i])

        # self.offset += num_epochs

    def predict(self, x_test):
        """ Predict, including scaling inputs/outputs """
        x_scaled = self.x_scaler.transform(x_test)
        y_scaled = self.model(x_scaled)
        y_test = self.y_scaler.inverse_transform(y_scaled)
        return y_test

    def calculate(self, x_test, diff=False):
        """ Predict with calculation of differentials or not """
        if diff:
            return self.calculate_with_greeks(x_test)
        else:
            return self.predict(x_test), None

    def calculate_with_greeks(self, x_test):
        """ Predit with calculation of differentials """
        # x-scaler in TF
        x_mean = self.x_scaler.mean_
        x_scale = self.x_scaler.scale_
        tf_x_mean = tf.convert_to_tensor(x_mean)
        tf_x_scale = tf.convert_to_tensor(x_scale)
        # y-scaler in TF
        y_mean = self.y_scaler.mean_
        y_scale = self.y_scaler.scale_
        tf_y_mean = tf.convert_to_tensor(y_mean, dtype='float32')
        tf_y_scale = tf.convert_to_tensor(y_scale, dtype='float32')

        # Evaluate and record differentials
        md_x_tensor = tf.convert_to_tensor(x_test)
        with tf.GradientTape() as t:
            t.watch(md_x_tensor)
            md_x_scaled = (md_x_tensor - tf_x_mean) / tf_x_scale
            md_y_scaled = self.model(md_x_scaled)
            md_y = md_y_scaled * tf_y_scale + tf_y_mean

        # Retrieve results
        base = md_y[0].numpy()
        # print(pv)
        grads = t.gradient(md_y, md_x_tensor)
        diffs = grads.numpy()
        # print(greeks)
        return base, diffs

    def save_to(self, path, root_file_name):
        """ Save to files """
        model_file, x_scaler_file, y_scaler_file = self.file_names(path, root_file_name)
        joblib.dump(self.x_scaler, x_scaler_file)
        joblib.dump(self.y_scaler, y_scaler_file)
        self.model.save(model_file)

    def convergence(self):
        """ Retrieve convergence history """
        return self.epochs, self.losses, self.accuracies

    def clear_training(self):
        """ Reset all training history """
        self.epochs.clear()
        self.losses.clear()
        self.accuracies.clear()
        self.offset = 0

    def scale_inputs(self, x_data):
        """ Scale inputs """
        return self.x_scaler.transform(x_data)

    def scaleback_outputs(self, y_data):
        """ Scale back outputs """
        return self.y_scaler.inverse_transform(y_data)

    @staticmethod
    def file_names(path, name):
        """ Specify file names for model save """
        root = path + "/" + name
        model_file = root + "_model.h5"
        x_scaler_file = root + "_xscaler.h5"
        y_scaler_file = root + "_yscaler.h5"
        return model_file, x_scaler_file, y_scaler_file


def read_learning_model(path, name):
    """ Load learning model from files """
    model_file, x_scaler_file, y_scaler_file = LearningModel.file_names(path, name)

    model = load_model(model_file)
    x_scaler = joblib.load(x_scaler_file)
    y_scaler = joblib.load(y_scaler_file)

    learning_model = LearningModel(model, x_scaler, y_scaler)

    return learning_model
