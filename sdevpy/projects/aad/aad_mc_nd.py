""" Calculate PV and Greeks for an arbitrary number of dimensions and compare performance
    of bumps vs AAD as the dimension increases. """
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


############### Runtime configuration #############################################################
DIM = 2
DTYPE = 'float64'
SEED = 42

tf.keras.backend.set_floatx(DTYPE)
rng = np.random.RandomState(SEED)

# Market data
RATE = 0.02
SPOT_MIN = 0.9
SPOT_MAX = 1.1
VOL_MIN = 0.15
VOL_MAX = 0.40
DIV_MIN = -0.01
DIV_MAX = 0.04

SPOT = rng.uniform(SPOT_MIN, SPOT_MAX, DIM).reshape(-1, 1)
VOL = rng.uniform(VOL_MIN, VOL_MAX, DIM).reshape(-1, 1)
DIV = rng.uniform(DIV_MIN, DIV_MAX, DIM).reshape(-1, 1)

# Payoff
STRIKE = 1.0

############### Bumps #############################################################################

############### AAD ###############################################################################

############### Numerical results #################################################################
