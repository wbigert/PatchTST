
import json
from models import PatchTST
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_Pred
import torch
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
class Configs:
      def __init__(self, **kwargs):
          for k, v in kwargs.items():
              setattr(self, k, v)

def get_model(run_path, model_config, transform=False):
    configs = Configs(**model_config)
    model = PatchTST.Model(configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(run_path + '/checkpoint.pth', map_location=device)
    if transform:
      state_dict = transform_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.double()
    model = model.to(device)
    model.eval()
    return model

def get_scaler(run_path):
    scaler_path = run_path + '/scaler.joblib'
    scaler = joblib.load(scaler_path)
    return scaler

def init(run_path, transform=False):
    # load dataset_loader_args.json
    with open(run_path + '/dataset_loader_args.json') as f:
        dataset_loader_args = json.load(f)
    
    # load model_config.json
    with open(run_path + '/model_config.json') as f:
        model_config = json.load(f)
    
    model = get_model(run_path, model_config, transform).float()
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

import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

def predict_trend(sequence, pred_len):
    """
    Predict future values based on linear trend of last few points in the sequence.
    
    Args:
    sequence: Input sequence of shape [1, seq_len, num_features].
    pred_len: Number of steps to predict into the future.
    
    Returns:
    A tensor of shape [1, pred_len, num_features] containing the predicted values.
    """
    num_points_to_use = sequence.size(1)  # Use last 10 points or less if not available
    x = np.arange(num_points_to_use).reshape(-1, 1)  # Time steps to use for prediction
    
    predictions = []
    for i in range(sequence.size(2)):  # Iterate over features
        y = sequence[0, -num_points_to_use:, i].cpu().numpy().reshape(-1, 1) 
        model = LinearRegression().fit(x, y)  # Fit linear model
        
        # Predict future values
        future_x = np.arange(num_points_to_use, num_points_to_use + pred_len).reshape(-1, 1)
        future_y = model.predict(future_x)
        
        predictions.append(future_y.flatten())
    
    prediction_tensor = torch.tensor(predictions, dtype=torch.float32).T  # Transpose to match expected shape
    prediction_tensor = prediction_tensor.unsqueeze(0)  # Add batch dimension
    return prediction_tensor

def inference(model, scaler, input_sequence, pred_len, ground_truth, col_names, trivial=False, verbose=False, device='cuda'):
    """
    input_sequence: [1, seq_len, num_features]
    ground_truth: [1, pred_len, num_features]
    """
    
    
    input_sequence = torch.as_tensor(input_sequence, dtype=torch.float32).to(device)
    ground_truth = torch.as_tensor(ground_truth, dtype=torch.float32).to(device)
    
    input_scaled = scale_sequence(scaler, input_sequence, device=device)
    ground_truth_scaled = scale_sequence(scaler, ground_truth, device=device)

    if trivial:
        prediction_scaled = predict_trend(input_scaled, pred_len)
    else:
        with torch.no_grad():
            prediction_scaled, _, _ = model(input_scaled)
            # Ensure prediction_scaled is [1, n_features, pred_len] if model's output doesn't match
            assert prediction_scaled.shape == (1, pred_len, input_sequence.shape[2] ), f"Model output shape mismatc. Expected: [1, {input_sequence.shape[2]}, {pred_len}], Got: {prediction_scaled.shape}"
    
    # Inverse scale predictions
    predictions_unscaled = scale_sequence(scaler, prediction_scaled, inverse=True, device=device)
    
    if verbose:
        # Compare unscaled and scaled predictions with corresponding ground truth
        compare_predictions(predictions_unscaled.squeeze(0), ground_truth.squeeze(0), col_names)
        compare_predictions(prediction_scaled.squeeze(0), ground_truth_scaled.squeeze(0), col_names)

    return to_numpy(prediction_scaled), to_numpy(predictions_unscaled), to_numpy(ground_truth_scaled), to_numpy(ground_truth)

# open csv file, grab a sequence of 50 rows, and the subsequent row as the ground truth
def sample_with_ground_truth(csv_data, seq_len, pred_len, anomaly=False):
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

# obtain all possible sequences of length seq_len, with a factor of 0.5 meaning half of all possible sequences are evenly obtained
def get_all_sequences(csv_data, seq_len, factor=0.5):
  column_names = csv_data.columns
  if 'date' in column_names:
    column_names = column_names.drop('date')
  column_names_list = column_names.tolist()
  sequences = []
  # use tqdm
  num_sequences = int(len(csv_data) * factor)
  indices = np.random.choice(len(csv_data) - seq_len + 1, num_sequences, replace=False)
  for i in tqdm(indices):
    sequence = csv_data.iloc[i:i + seq_len][column_names_list].to_numpy()
    sequence = torch.tensor(sequence, dtype=torch.float32, device='cuda')  # Convert to float32 for compatibility with PyTorch models
    sequence = sequence.unsqueeze(0)  # [1, seq_len, num_features]
    sequences.append(sequence)
  return sequences

def get_sequences(csv_data, n, seq_len, mode='normal'):
    ANOM_LENGTH = 10
    column_names = csv_data.columns
    if 'date' in column_names:
        csv_data = csv_data.drop(columns=['date'])
    column_names_list = csv_data.columns.tolist()
    
    sequences = []
    
    if mode == 'random':
        # For each of the n sequences, select seq_len random rows
        indices = np.random.randint(0, len(csv_data), (n, seq_len))
        sequences = csv_data.iloc[indices.flatten()].to_numpy().reshape(n, seq_len, len(column_names_list))
    elif mode == 'normal':
        # Generate n random start points
        starts = np.random.randint(0, len(csv_data) - seq_len, n)
        sequences = np.array([csv_data.iloc[start:start + seq_len][column_names_list].to_numpy() for start in starts])
    elif mode == 'anomalous':
        # Generate n random start points
        starts = np.random.randint(0, len(csv_data) - seq_len, n)
        for start in starts:
            sequence = csv_data.iloc[start:start + seq_len][column_names_list].to_numpy()
            random_starts = np.random.randint(0, len(csv_data) - ANOM_LENGTH, ANOM_LENGTH)
            sequence[-ANOM_LENGTH:] = np.array([csv_data.iloc[random_start][column_names_list].to_numpy() for random_start in random_starts])
            sequences.append(sequence)
    else:
        return None
    
    return np.array(sequences), column_names_list


def transform_state_dict(old_state_dict):
    # Transform the keys from the old state dict format to the new one
    new_state_dict = {}
    for key in old_state_dict:
        new_key = key
        # Replace 'ff.0' with 'ff1' and 'ff.3' with 'ff2'
        if 'ff.0' in key:
            new_key = key.replace('ff.0', 'ff1')
        elif 'ff.3' in key:
            new_key = key.replace('ff.3', 'ff2')
        new_state_dict[new_key] = old_state_dict[key]
    return new_state_dict

def explore_centroids(flattened_embeddings, k_range=range(1, 20)):
      inertias = []

      # Wrap k_range with tqdm for a progress bar
      for k in tqdm(k_range, desc='Fitting KMeans'):
          kmeans = KMeans(n_clusters=k, random_state=0).fit(flattened_embeddings)
          inertias.append(kmeans.inertia_)

      plt.figure(figsize=(8, 5))
      plt.plot(k_range, inertias, 'bx-')
      plt.xlabel('k (number of clusters)')
      plt.ylabel('Inertia')
      plt.title('Elbow Method for Optimal k')
      plt.show()

def check_saved(file, run_path):
    return os.path.isfile(f'{run_path}/{file}')

def hist_plot(save_path, data, x, hue, element, bins, title, xlabel, ylabel):
  plt.figure(figsize=(12, 6))
  sns.histplot(data=data, x=x, hue=hue, element=element, bins=bins)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.savefig(save_path)