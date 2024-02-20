import os
import subprocess
import pandas as pd
import json

# This functions runs the supervised version of PatchTST on a custom dataset 
# and saves the resulting dataset loader args and model parameters so that
# the resulting model can easily be instantiated again and used for inference.
# PatchTST has been modified to also save the scaler after training, so that
# the model can be used for inference without having to retrain the scaler.
def run_custom(pred_len=1, seq_len=50, train_epochs=100, patch_len=16, d_model=128, d_ff=256, data_name='weather', root_path='./data/', model='PatchTST'):
  data_path = data_name + ".csv"
  model_id = f"model_{model}_name_{data_name}_seqlen_{seq_len}_predlen_{pred_len}_epochs_{train_epochs}_patchlen_{patch_len}_dmodel_{d_model}_dff_{d_ff}"
  run_path = root_path + "runs/" + model_id + "/"

  csv_data = pd.read_csv(os.path.join(root_path, data_path))
  column_names = csv_data.columns
  if 'date' in column_names:
      column_names = column_names.drop('date')

  column_names_list = column_names.tolist()

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

  model_config = {
      "enc_in": 21,
      "seq_len": seq_len,
      "pred_len": pred_len,
      "e_layers": 3,
      "n_heads": 16,
      "d_model": d_model,
      "d_ff": d_ff,
      "dropout": 0.2,
      "fc_dropout": 0.2,
      "head_dropout": 0.0,
      "patch_len": patch_len,
      "stride": 8,
      "padding_patch": 'end',
      "revin": 1,
      "affine": False,
      "subtract_last": False,
      "decomposition": False,
      "kernel_size": 25,
      "individual": False,
      "features": "M",
      "label_len": 0,
      "model": model,
  }

  command = f"python -u ./run_longExp.py " \
            f"--random_seed 2021 " \
            f"--is_training 1 " \
            f"--root_path {root_path} " \
            f"--data_path {data_path} " \
            f"--model_id {model_id} " \
            f"--model {model} " \
            f"--data custom " \
            f"--features M " \
            f"--seq_len {seq_len} " \
            f"--pred_len {pred_len} " \
            f"--label_len {0} " \
            f"--enc_in 21 " \
            f"--e_layers 3 " \
            f"--n_heads 16 " \
            f"--d_model {d_model} " \
            f"--d_ff {d_ff} " \
            f"--dropout 0.2 " \
            f"--fc_dropout 0.2 " \
            f"--head_dropout 0 " \
            f"--patch_len {patch_len} " \
            f"--stride 8 " \
            f"--des 'Exp' " \
            f"--train_epochs {train_epochs} " \
            f"--patience 20 " \
            f"--itr 1 " \
            f"--batch_size 128 " \
            f"--learning_rate 0.0001 " \
            f"--freq min " \
            f"--checkpoints {run_path} " \

  # Log files for stdout and stderr
  stdout_log_path = os.path.join(run_path, "stdout.log")
  stderr_log_path = os.path.join(run_path, "stderr.log")
  print(f"Executing command: {command}")

  # Make sure the checkpoints directory exists
  os.makedirs(run_path, exist_ok=True)

  # Store dataset_loader_args to checkpoints directory as a json file
  with open(os.path.join(run_path, "dataset_loader_args.json"), 'w') as f:
      json.dump(dataset_loader_args, f)

  # Store model_config to checkpoints directory as a json file
  with open(os.path.join(run_path, "model_config.json"), 'w') as f:
      json.dump(model_config, f)

  with open(stdout_log_path, 'w') as stdout_file, open(stderr_log_path, 'w') as stderr_file:
      subprocess.run(command, shell=True, stdout=stdout_file, stderr=stderr_file)

  # After subprocess.run() completes, return the run_path
  return run_path


if __name__ == '__main__':
    run_paths = []
    # for d_model in [16, 32, 48, 64, 80, 96, 112, 128, 144, 160]:
    for d_model in [128]:
        run_path = run_custom(d_model=d_model, d_ff=d_model*2, train_epochs=50, patch_len=16, model='PatchTST')
        run_paths.append(run_path)
        print(f"Completed run for d_model={d_model} with run_path: {run_path}")
    
    best_score = float('inf')
    best_run_path = ""
    for run_path in run_paths:
        with open(os.path.join(run_path, "results.txt")) as f:
            results = f.read()
            mse = float(results.split(",")[0].split(":")[1])
            print(f"mse for {run_path}: {mse}")
            if mse < best_score:
                best_score = mse
                best_run_path = run_path

    print(f"The best run was {best_run_path} with mse: {best_score}")
