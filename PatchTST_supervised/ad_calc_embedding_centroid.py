from ad_utils import init, get_all_sequences, scale_sequence, explore_centroids, check_saved
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def process_embeddings(run_path, explore_centroids=False, K=10, N=20000):
  model, scaler, dataset_loader_args, model_config = init(run_path, transform=True)
  enc_in, seq_len, d_model, d_ff = model_config['enc_in'], model_config['seq_len'], model_config['d_model'], model_config['d_ff']
  csv_data = pd.read_csv(dataset_loader_args['root_path'] + dataset_loader_args['data_path'])
  # csv_data = pd.read_csv('./inference_input.csv')
  already_saved = check_saved('sequences.pt', run_path)
  if not already_saved:
    print("Sequences not found, calculating new...")
    sequences = get_all_sequences(csv_data, model_config['seq_len'], 0.5)
    torch.save(sequences, f'{run_path}/sequences.pt')
  else:
    print("Loading sequences...")
    sequences = torch.load(f'{run_path}/sequences.pt')

  already_saved = check_saved('all_embeddings.npy', run_path) or check_saved('all_high_dim_embeddings.npy', run_path)
  if already_saved:
    print("Loading all_embeddings and all_high_dim_embeddings...")
    all_embeddings = np.load(f'{run_path}/all_embeddings.npy')
    all_high_dim_embeddings = np.load(f'{run_path}/all_high_dim_embeddings.npy')
  else:
    print("Embeddings not found, calculating new...")
    for i, sequence in enumerate(tqdm(sequences)):
      sequence_scaled = scale_sequence(scaler, sequence, device='cuda')
      _, embeddings, high_dim_embeddings = model(sequence)
      if i == 0:
        num_patches = embeddings.shape[-1]
        all_high_dim_embeddings = np.zeros((len(sequences), enc_in, num_patches, d_ff)) # high_dim has shape [17, 6, 256] [n_vars, num_patches, d_ff]
        all_embeddings = np.zeros((len(sequences), enc_in, d_model, num_patches))
      all_embeddings[i] = embeddings.detach().cpu().numpy()
      all_high_dim_embeddings[i] = high_dim_embeddings.detach().cpu().numpy()
  
  # save all_embeddings and all_high_dim_embeddings
  if not already_saved:
    print("Saving all_embeddings and all_high_dim_embeddings...")
    np.save(f'{run_path}/all_embeddings.npy', all_embeddings)
    np.save(f'{run_path}/all_high_dim_embeddings.npy', all_high_dim_embeddings)

  # Flatten the embeddings to fit KMeans which expects 2D input: (n_samples, n_features)
  flattened_embeddings = all_embeddings.reshape(len(sequences), -1)
  flattened_high_dim_embeddings = all_high_dim_embeddings.reshape(len(sequences), -1)

  # Perform k-means clustering on the flattened embeddings, use elbow method to determine the number of clusters
  if explore_centroids:
    print("Exploring centroids...")
    explore_centroids(flattened_embeddings)

  if not check_saved(f'kmeans_k_{K}.pkl', run_path) and not check_saved(f'kmeans_high_dim_k_{K}.pkl', run_path):
    print("K-means model not found, creating new model...")
    kmeans = KMeans(n_clusters=K, random_state=0).fit(flattened_embeddings)
    kmeans_high_dim = KMeans(n_clusters=K, random_state=0).fit(flattened_high_dim_embeddings)

    # Save the k-means model using joblib
    joblib.dump(kmeans, f'{run_path}/kmeans_k_{K}.pkl')
    joblib.dump(kmeans_high_dim, f'{run_path}/kmeans_high_dim_k_{K}.pkl')
    print("K-means model saved.")


  if not check_saved('centroid.npy', run_path) and not check_saved('centroid_high_dim.npy', run_path):
    print("Centroids not found, calculating new...")
    centroid = np.mean(all_embeddings, axis=0)
    centroid_high_dim = np.mean(all_high_dim_embeddings, axis=0)

    np.save(f'{run_path}/centroid.npy', centroid)
    np.save(f'{run_path}/centroid_high_dim.npy', centroid_high_dim)
    print("Centroids saved.")
  return kmeans, kmeans_high_dim, centroid, centroid_high_dim

    
if __name__ == '__main__':
  _, _, _, _ = process_embeddings('./data/runs/model_PatchTST_name_weather_seqlen_50_predlen_1_epochs_50_patchlen_16_dmodel_128_dff_256', explore_centroids=False, K=10, N=20000)
  print("Done")