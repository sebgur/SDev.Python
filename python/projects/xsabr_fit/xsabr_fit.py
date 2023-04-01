import settings
import os
import pandas as pd
from projects.xsabr_fit.HaganSabrGenerator import HaganSabrGenerator

# ################ ToDo ###################################################################################
# Read the dataframe from the generator. The generator knows the expected structure, since that
# structure has been created by it.
# Implement generic cleansing and conversion from prices
# Implement generic training

# Retrieve data from generated file
model_type = 'HaganSABR'
data_folder = os.path.join(settings.workfolder, "XSABRSamples")
data_file = os.path.join(data_folder, model_type + "_samples.tsv")
data_df = pd.read_csv(data_file, sep='\t')
print(data_df)

generator = HaganSabrGenerator(shift)

# Cleanse data
# We take the prices and attempt to convert them to normal volatilities. Any failure to convert
# is removed from the dataset. We also remove all vols that are not in the acceptable range.
def clean_price_to_nvol(data_df):
    # data = data.drop(data[data.IV > 1.5].index)  # Remove high vols
    return prices

