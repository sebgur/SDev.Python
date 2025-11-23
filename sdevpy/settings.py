""" Global runtime settings such as workfolder path, warning configurations, etc. """
import os
from silence_tensorflow import silence_tensorflow
# import clr # For pythonnet
# from System import *

# # Global variables
# WORKFOLDER = r"C:\temp\sdevpy"

# VEGA_SCALING = 0.01  # Vega shown for 1% absolute moves
# THETA_SCALING = 1.0 / 365.0  # Theta shown for 1d moves
# DV01_SCALING = 1.0 / 10000.0  # DV01 shown for 1bp moves
# VOLGA_SCALING = VEGA_SCALING**2
# VANNA_SCALING = VEGA_SCALING

# # Disable debug warnings in tensorflow
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # To silence tensorflow's constant nagging
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # To silence tensorflow's constant nagging
# silence_tensorflow() # This one works best at least in tf 2.15.0

# # Python.net
# BIN_PATH = "C:\\Code\\SDev\\Excel\\SDev.Addin\\bin\\x64\\Release\\"
# USERNAME = Environment.UserName # Example usage of C# System

# # # Add .config file
# # # config_file = "my_path_to_.config_file"
# # # AppDomain.CurrentDomain.SetData("APP_CONFIG_FILE", config_file)

# # Load dlls
# def load_dll(bin_path, dll_name):
#     path = r'%s%s' % (bin_path, dll_name)
#     clr.AddReference(path)

# # Doing nothing for now, just to avoid warning of dummy import
# def apply_settings():
#     """ Dummy method to apply settings when necessary """
#     return 0


# load_dll(BIN_PATH, 'SDev.Addin')

# from SDev.Addin import *
# from SDev.Addin import Loader as sdev
# # from SDev.Addin.WorksheetFunctions import *
# from SDev.Addin import WorksheetFunctions as wf

# if __name__ == "__main__":
#     print("Hello " + USERNAME)
#     # print(sdev.sdHelloWorld())

#     iv = 0.20
#     fwd = 0.04
#     strike = 0.03
#     maturity = 2.5
#     is_call = True
#     price = wf.xlBachelier.sdBachelierPrice(fwd, strike, iv, is_call)
#     print(price)