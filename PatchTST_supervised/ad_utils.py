
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

def to_numpy(tensor, squeeze=True):
    if squeeze:
        return tensor.detach().cpu().numpy().squeeze()
    return tensor.detach().cpu().numpy()

def inference(model, scaler, input_sequence, ground_truth, col_names, trivial=False, verbose=False, device='cuda'):
    # Convert and move input and ground truth to the correct device at the start
    input_sequence = torch.as_tensor(input_sequence, dtype=torch.float32).to(device)
    ground_truth = torch.as_tensor(ground_truth, dtype=torch.float32).to(device)
    
    # Scale input and ground_truth
    input_scaled = scale_sequence(scaler, input_sequence, device=device)
    ground_truth_scaled = scale_sequence(scaler, ground_truth, device=device)

    with torch.no_grad():
        prediction_scaled, embeddings = model(input_scaled)
    
    # Use latest time step in input sequence as the prediction, instead of the prediction from the model
    if trivial:
        prediction_scaled = input_scaled[:, -1:]
    else:
        with torch.no_grad():
          prediction_scaled, embeddings = model(input_scaled)
    verbose and print(f"shape embeddings: {embeddings.shape}")
    # Inverse scale predictions
    predictions_unscaled = scale_sequence(scaler, prediction_scaled, inverse=True, device=device)
    
    if verbose:
        # Compare unscaled and scaled predictions with corresponding ground truth
        compare_predictions(predictions_unscaled.squeeze(0), ground_truth.squeeze(0), col_names)
        compare_predictions(prediction_scaled.squeeze(0), ground_truth_scaled.squeeze(0), col_names)

    return to_numpy(prediction_scaled), to_numpy(predictions_unscaled), to_numpy(ground_truth_scaled), to_numpy(ground_truth)

# open csv file, grab a sequence of 50 rows, and the subsequent row as the ground truth
def take_random_sample(csv_data, seq_len, pred_len, anomaly=False):
    column_names = csv_data.columns
    if 'date' in column_names:
        column_names = column_names.drop('date')
    column_names_list = column_names.tolist()

    # Ensure there's at least 1 valid starting point
    max_start_index = max(0, len(csv_data) - seq_len - pred_len)
    start = np.random.randint(0, max_start_index + 1)  # +1 to include the last possible index
    end = start + seq_len
    sequence = csv_data.iloc[start:end][column_names_list].to_numpy()

    if anomaly:
        # Exclude the sequence range and ensure the selected anomaly step is not within it
        valid_indices = [i for i in range(len(csv_data)) if i < start or i >= end + pred_len]
        anomaly_index = np.random.choice(valid_indices)  # Removed the '1' to get a scalar directly
        # Ensure anomaly_index does not exceed the dataframe's bounds for the requested pred_len
        max_anomaly_index = max(0, len(csv_data) - pred_len)
        anomaly_index = min(anomaly_index, max_anomaly_index)
        ground_truth = csv_data.iloc[anomaly_index:anomaly_index + pred_len][column_names_list].to_numpy()
    else:
        ground_truth = csv_data.iloc[end:end + pred_len][column_names_list].to_numpy()

    sequence = torch.tensor(sequence, dtype=torch.float32)  # Convert to float32 for compatibility with PyTorch models
    sequence = sequence.unsqueeze(0)  # [1, seq_len, num_features]
    ground_truth = torch.tensor(ground_truth, dtype=torch.float32)  # [pred_len, num_features]
    ground_truth = ground_truth.unsqueeze(0)  # [1, pred_len, num_features]

    return sequence, ground_truth, column_names_list


