import tools.sabr as sabr

model_name = 'Hagan_SABR_vec'
scaler_file = 'outputs/' + model_name + '_scaler.h5'
model_file = 'outputs/' + model_name + '_model.h5'

sabr.view_sabr_smiles_vec(model_file, scaler_file)
