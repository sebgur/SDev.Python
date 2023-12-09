""" Global runtime settings such as workfolder path, warning configurations, etc. """
import os
from silence_tensorflow import silence_tensorflow

# Global variables
WORKFOLDER = r"C:\temp\sdevpy"

VEGA_SCALING = 0.01  # Vega shown for 1% absolute moves
THETA_SCALING = 1.0 / 365.0  # Theta shown for 1d moves
DV01_SCALING = 1.0 / 10000.0  # DV01 shown for 1bp moves
VOLGA_SCALING = VEGA_SCALING**2
VANNA_SCALING = VEGA_SCALING

# Disable debug warnings in tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # To silence tensorflow's constant nagging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # To silence tensorflow's constant nagging
silence_tensorflow() # This one works best at least in tf 2.15.0

# Doing nothing for now, just to avoid warning of dummy import
def apply_settings():
    """ Dummy method to apply settings when necessary """
    return 0
