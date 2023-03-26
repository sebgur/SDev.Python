import tensorflow as tf


# Saving/loading weights
model.save_weights("model_current_best.h5")
model.load_weights("model_current_best.h5")

# Piecewise decay of the learning rate: the first 1000 steps use a learning rate of 0.01,
# from 5,000 - 10,000: learning rate = 0.001
# from 10,000 onwards: learning rate = 0.0005
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([5000, 10000], [1e-2, 1e-3, 5e-4])

# Choose the optimizer
optim = tf.keras.optimizers.Adam(learning_rate=lr)

kinits = tf.keras.initializers
klayers = tf.keras.layers

init = kinits.glorot_normal

# init = kinit.RandomNormal(mean=0.0, stddev=1.0 / np.sqrt(num_neurons), seed=42)

# model.add(Dropout(0.1))


# Strip TensorFlow's .jacobian() into a series of Jacobian matrices per point
def strip_jacobian(jacobian):
    shape = jacobian.shape
    num_points = shape[0]
    dim = shape[1]
    stripped_jacobian = []
    for i in range(num_points):
        jac = jacobian[i]
        new_jac = []
        for j in range(dim):
            new_jac.append(jac[j][i].numpy())

        stripped_jacobian.append(new_jac)

    return stripped_jacobian
