
import json
from models import PatchTST
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_Pred
import torch
import joblib
import pandas as pd
import numpy as np
class Configs:
      def __init__(self, **kwargs):
          for k, v in kwargs.items():
              setattr(self, k, v)

def get_model(run_path, model_config):
    configs = Configs(**model_config)
    model = PatchTST.Model(configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(run_path + '/checkpoint.pth', map_location=device))
    model.double()
    model = model.to(device)
    model.eval()
    return model

def get_scaler(run_path):
    scaler_path = run_path + '/scaler.joblib'
    scaler = joblib.load(scaler_path)
    return scaler

def init(run_path):
    # load dataset_loader_args.json
    with open(run_path + '/dataset_loader_args.json') as f:
        dataset_loader_args = json.load(f)
    
    # load model_config.json
    with open(run_path + '/model_config.json') as f:
        model_config = json.load(f)
    
    model = get_model(run_path, model_config).float()
    scaler = get_scaler(run_path)

    return model, scaler, dataset_loader_args, model_config

def scale_sequence(scaler, sequence, inverse=False, device='cuda'):
    # Ensure sequence is a PyTorch tensor on the correct device before converting to numpy
    sequence = torch.as_tensor(sequence, dtype=torch.float32).to(device)
    # Squeeze and convert to CPU numpy for scaling
    sequence_np = sequence.squeeze(0).cpu().numpy()
    sequence_scaled_np = scaler.transform(sequence_np) if not inverse else scaler.inverse_transform(sequence_np)
    # Convert back to PyTorch tensor and ensure it's on the correct device
    sequence_scaled = torch.tensor(sequence_scaled_np, dtype=torch.float32, device=device).unsqueeze(0)
    return sequence_scaled

def compare_predictions(predictions, ground_truth, col_names):
    for i, name in enumerate(col_names):
        diff = predictions[:, i] - ground_truth[:, i]
        print(f"Feature: {name}, diff={diff.mean().item()}, Prediction: {predictions[:, i].mean().item()}, Ground truth: {ground_truth[:, i].mean().item()}")

def inference(model, scaler, input_sequence, ground_truth, col_names, verbose=False, device='cuda'):
    # Convert and move input and ground truth to the correct device at the start
    input_sequence = torch.as_tensor(input_sequence, dtype=torch.float32).to(device)
    ground_truth = torch.as_tensor(ground_truth, dtype=torch.float32).to(device)
    
    # Scale input and ground_truth
    input_scaled = scale_sequence(scaler, input_sequence, device=device)
    ground_truth_scaled = scale_sequence(scaler, ground_truth, device=device)

    with torch.no_grad():
        prediction_scaled = model(input_scaled)

    # Inverse scale predictions
    predictions_unscaled = scale_sequence(scaler, prediction_scaled, inverse=True, device=device)
    
    if verbose:
        # Compare unscaled and scaled predictions with corresponding ground truth
        compare_predictions(predictions_unscaled.squeeze(0), ground_truth.squeeze(0), col_names, verbose)
        compare_predictions(prediction_scaled.squeeze(0), ground_truth_scaled.squeeze(0), col_names, verbose)

    return prediction_scaled.squeeze(0).cpu().numpy(), predictions_unscaled.squeeze(0).cpu().numpy(), ground_truth_scaled.squeeze(0).cpu().numpy(), ground_truth.squeeze(0).cpu().numpy()

# open csv file, grab a sequence of 50 rows, and the subsequent row as the ground truth
def take_random_sample(csv_data, seq_len, pred_len):
    column_names = csv_data.columns
    if 'date' in column_names:
        column_names = column_names.drop('date')
    column_names_list = column_names.tolist()

    # Ensure there's at least 1 valid starting point
    max_start_index = max(0, len(csv_data) - seq_len - pred_len)
    start = np.random.randint(0, max_start_index + 1)  # +1 to include the last possible index
    end = start + seq_len
    sequence = csv_data.iloc[start:end][column_names_list].to_numpy()
    ground_truth = csv_data.iloc[end:end + pred_len][column_names_list].to_numpy()

    sequence = torch.tensor(sequence, dtype=torch.float64)  # [seq_len, num_features]
    sequence = sequence.unsqueeze(0)  # [1, seq_len, num_features]
    ground_truth = torch.tensor(ground_truth, dtype=torch.float64)  # [pred_len, num_features]
    ground_truth = ground_truth.unsqueeze(0)  # [1, pred_len, num_features]

    return sequence, ground_truth, column_names_list


if __name__ == '__main__':
    run_path = './data/runs/name_weather_seqlen_50_predlen_1_epochs_100_patchlen_16_dmodel_128_dff_256'
    model, scaler, dataset_loader_args, model_config = init(run_path)

    csv_data = pd.read_csv(dataset_loader_args['root_path'] + dataset_loader_args['data_path'])
    # csv_data = pd.read_csv('./inference_input.csv')
    verbose = False
    N = 1000

    preds_scaled_list = np.zeros((N, model_config['pred_len'], model_config['enc_in']))
    preds_raw_list = np.zeros((N, model_config['pred_len'], model_config['enc_in']))
    trues_scaled_list = np.zeros((N, model_config['pred_len'], model_config['enc_in']))
    trues_raw_list = np.zeros((N, model_config['pred_len'], model_config['enc_in']))

    for i in range(N):
      if i % 1000 == 0:
        print(f"i: {i}")
      sequence, ground_truth, column_names_list = take_random_sample(csv_data, model_config['seq_len'], model_config['pred_len'])
      preds_scaled, preds_raw, trues_scaled, trues_raw = inference(model, scaler, sequence, ground_truth, column_names_list, verbose=verbose)
      preds_scaled_list[i] = preds_scaled
      preds_raw_list[i] = preds_raw
      trues_scaled_list[i] = trues_scaled
      trues_raw_list[i] = trues_raw

    mean_MSE_raw = np.mean((preds_raw_list - trues_raw_list) ** 2)
    mean_MSE_scaled = np.mean((preds_scaled_list - trues_scaled_list) ** 2)

    print(f"mean_MSE_raw: {mean_MSE_raw}")
    print(f"mean_MSE_scaled: {mean_MSE_scaled}")