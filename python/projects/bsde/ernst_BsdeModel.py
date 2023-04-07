import tensorflow as tf
import numpy as np


# class MyClass:
#     def __init__(self, **kwargs):
#         print('alhpahs;dfkljas ')
#         self.dim = kwargs.get('dim', 2)
#         print(self.dim)


class BsdeModel(tf.keras.Model):
    def __init__(self, dtype='float32', dim=1, num_steps=1, **kwargs):
        # Call initializer of tf.keras.Model
        super().__init__(**kwargs)

        self.ftype = dtype
        self.dim = dim
        self.num_steps = num_steps

        # Initialize the value u(0, x) randomly
        u0 = np.random.uniform(.1, .3, size=1).astype(self.dtype)
        self.u0 = tf.Variable(u0)

        # Initialize the gradient nabla u(0, x) randomly
        gradu0 = np.random.uniform(-1e-1, 1e-1, size=(1, self.dim)).astype(self.ftype)
        self.gradu0 = tf.Variable(gradu0)

        # Create template of dense layer without bias and activation
        _dense = lambda dim: tf.keras.layers.Dense(
            units=dim,
            activation=None,
            use_bias=False)

        # Create template of batch normalization layer
        _bn = lambda: tf.keras.layers.BatchNormalization(
            momentum=.99,
            epsilon=1e-6,
            beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
            gamma_initializer=tf.random_uniform_initializer(0.1, 0.5))

        # Initialize a list of networks approximating the gradient of u(t, x) at t_i
        self.gradui = []

        # Loop over number of time steps
        for _ in range(self.num_steps - 1):
            # Batch normalization on dim-dimensional input
            this_grad = tf.keras.Sequential()
            this_grad.add(tf.keras.layers.Input(self.dim))
            this_grad.add(_bn())

            # Two hidden layers of type (Dense -> Batch Normalization -> ReLU)
            for _ in range(2):
                this_grad.add(_dense(self.dim + 10))
                this_grad.add(_bn())
                this_grad.add(tf.keras.layers.ReLU())

            # Dense layer followed by batch normalization for output
            this_grad.add(_dense(self.dim))
            this_grad.add(_bn())
            self.gradui.append(this_grad)
