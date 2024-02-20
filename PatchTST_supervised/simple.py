from ad_utils import init, inference, take_random_sample
import pandas as pd
import numpy as np
import torch

def save_with_column_names(filename, data, column_names):
    with open(filename, 'w') as f:
        for col_name, value in zip(column_names, data.flatten()):
            # Ensure full precision is printed and format as 'column_name: value'
            f.write(f"{col_name}: {value:.16f}\n")  # Adjust the precision as needed

if __name__ == '__main__':
    run_path = './data/runs/model_PatchTST_name_weather_seqlen_50_predlen_1_epochs_50_patchlen_16_dmodel_128_dff_256'
    model, scaler, dataset_loader_args, model_config = init(run_path)

    print('scaler info')
    print(scaler.mean_)
    print(scaler.scale_)
    print(scaler.var_)
    print(scaler.n_samples_seen_)
    print(model.print_model_parameters())
    batch_x = torch.load('vali_batch_x.pt')
    batch_y = torch.load('vali_batch_y.pt')
    outputs, _ = model(batch_x)

    column_names = ['p (mbar)','T (degC)','Tpot (K)','Tdew (degC)','rh (%)','VPmax (mbar)','VPact (mbar)','VPdef (mbar)','sh (g/kg)','H2OC (mmol/mol)','rho (g/m**3)','wv (m/s)','max. wv (m/s)','wd (deg)','rain (mm)','raining (s)','SWDR (W/m²)','PAR (µmol/m²/s)','max. PAR (µmol/m²/s)','Tlog (degC)','OT']

    inputs_inverse_scaled = scaler.inverse_transform(batch_x[0].cpu().detach().numpy())
    outputs_inverse_scaled = scaler.inverse_transform(outputs[0].cpu().detach().numpy())
    actual_inverse_scaled = scaler.inverse_transform(batch_y[0].cpu().detach().numpy())

    # save entire batch_x and batch_y to file
    torch.save(batch_x, 'vali_batch_x.pt')
    torch.save(batch_y, 'vali_batch_y.pt')

    save_with_column_names('simple_actual.txt', actual_inverse_scaled, column_names)
    print('simple_actual.txt saved')
    save_with_column_names('simple_pred.txt', outputs_inverse_scaled, column_names)
    print('simple_pred.txt saved')
    inputs_inverse_scaled = np.vstack([column_names, inputs_inverse_scaled])
    inputs_inverse_scaled = np.vstack([inputs_inverse_scaled, actual_inverse_scaled])
    np.savetxt('simple_input.csv', inputs_inverse_scaled, delimiter=',', fmt='%s')
    print('simple_input.csv saved')