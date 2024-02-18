import os
import subprocess
import pandas as pd
import json

# Only for debugging: Test with a single prediction length
pred_len = 1  # Example prediction length for testing
seq_len = 50
train_epochs = 5
model_id = f"weather_{seq_len}_{pred_len}"
root_path = "./data/"
checkpoints = root_path + "runs/" + model_id + "/"
data_path = "weather.csv"

# The none date columns of the dataset specified in the data_path, e.g. weather.csv
csv_data = pd.read_csv(os.path.join(root_path, data_path))
column_names = csv_data.columns
if 'date' in column_names:
    column_names = column_names.drop('date')

column_names_list = column_names.tolist()
print(f"Columns in the dataset: {column_names_list}")

dataset_loader_args = {
    'root_path': root_path,
    'data_path': data_path,
    'flag': 'pred',
    'size': [seq_len, 0, pred_len],
    'features': 'M',
    'cols': column_names_list,
    'timeenc': 0, # irrelevant, patchTST does not use this
    'freq': 'min', # irrelevant, patchTST does not use this
    'target': 'OT' # irrelevant, patchTST does not use this 
}

patchTST_model_config = {
    "enc_in": 21,
    "seq_len": seq_len,
    "pred_len": pred_len,
    "e_layers": 3,
    "n_heads": 16,
    "d_model": 128,
    "d_ff": 256,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "head_dropout": 0.0,
    "patch_len": 16,
    "stride": 8,
    "padding_patch": 'end',
    "revin": False,
    "affine": False,
    "subtract_last": False,
    "decomposition": False,
    "kernel_size": 25,
    "individual": False,
    "features": "M",
    "label_len": 0,
}

command = f"python -u ./run_longExp.py " \
          f"--random_seed 2021 " \
          f"--is_training 1 " \
          f"--root_path {root_path} " \
          f"--data_path {data_path} " \
          f"--model_id {model_id} " \
          f"--model PatchTST " \
          f"--data custom " \
          f"--features M " \
          f"--seq_len {seq_len} " \
          f"--pred_len {pred_len} " \
          f"--label_len {0} " \
          f"--enc_in 21 " \
          f"--e_layers 3 " \
          f"--n_heads 16 " \
          f"--d_model 128 " \
          f"--d_ff 256 " \
          f"--dropout 0.2 " \
          f"--fc_dropout 0.2 " \
          f"--head_dropout 0 " \
          f"--patch_len 16 " \
          f"--stride 8 " \
          f"--des 'Exp' " \
          f"--train_epochs {train_epochs} " \
          f"--patience 20 " \
          f"--itr 1 " \
          f"--batch_size 128 " \
          f"--learning_rate 0.0001 " \
          f"--freq min " \
          f"--checkpoints {checkpoints} " \

# Directory where you want to store the logs

# Log files for stdout and stderr
stdout_log_path = os.path.join(checkpoints, "stdout.log")
stderr_log_path = os.path.join(checkpoints, "stderr.log")
print(f"Executing command: {command}")

# Make sure the checkpoints directory exists
os.makedirs(checkpoints, exist_ok=True)

# Store dataset_loader_args to checkpoints directory as a json file
with open(os.path.join(checkpoints, "dataset_loader_args.json"), 'w') as f:
    json.dump(dataset_loader_args, f)

# Store model_config to checkpoints directory as a json file
with open(os.path.join(checkpoints, "model_config.json"), 'w') as f:
    json.dump(patchTST_model_config, f)

with open(stdout_log_path, 'w') as stdout_file, open(stderr_log_path, 'w') as stderr_file:
    subprocess.run(command, shell=True, stdout=stdout_file, stderr=stderr_file)


