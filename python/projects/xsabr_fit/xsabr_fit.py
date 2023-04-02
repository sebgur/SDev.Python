import settings
import os
import pandas as pd
import numpy as np
from projects.xsabr_fit.HaganSabrGenerator import HaganSabrGenerator, ShiftedHaganSabrGenerator

# ################ ToDo ###################################################################################
# Implement generic training
# Display result and metrics

# ################ Runtime configuration ##################################################################
data_folder = os.path.join(settings.workfolder, "XSABRSamples")
model_type = "ShiftedHaganSABR"

# Generator factory
if model_type == "HaganSABR":
    generator = HaganSabrGenerator()
elif model_type == "ShiftedHaganSABR":
    generator = ShiftedHaganSabrGenerator()
else:
    raise Exception("Unknown model: " + model_type)


# Retrieve dataset
data_file = os.path.join(data_folder, model_type + "_samples.tsv")
print("Reading data from file: " + data_file)
x_set, y_set, data_df = generator.retrieve_datasets(data_file)
print(data_df.head())

# Training


# print(x_set)
# print(y_set.shape)
# print(y_set)

