""" Global runtime settings such as workfolder path, warning configurations, etc. """
import os

# Global variables
WORKFOLDER = r"C:\temp\sdevpy"

# Disable debug warnings in tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Doing nothing for now, just to avoid warning of dummy import
def apply_settings():
    """ Dummy method to apply settings when necessary """
    return 0
