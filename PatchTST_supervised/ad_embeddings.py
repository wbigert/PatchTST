import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from ad_utils import init, get_sequences, scale_sequence, check_saved, hist_plot
import joblib
import tqdm
# Initialization
if __name__ == '__main__':
  run_path = './data/runs/model_PatchTST_name_weather_trimmed_seqlen_50_predlen_12_epochs_50_patchlen_16_dmodel_128_dff_256'
  model, scaler, dataset_loader_args, model_config = init(run_path, transform=True)
  
  if not check_saved('centroid_high_dim.npy', run_path) or not check_saved('centroid.npy', run_path):
    print("Centroid not found, calculate them first.")
    quit()
    
  centroid = torch.as_tensor(np.load(f'{run_path}/centroid.npy'), dtype=torch.float32).to('cuda')
  centroid_high_dim = torch.as_tensor(np.load(f'{run_path}/centroid_high_dim.npy'), dtype=torch.float32).to('cuda')
  csv_data = pd.read_csv(dataset_loader_args['root_path'] + dataset_loader_args['data_path'])
  
  if not check_saved('kmeans.pkl', run_path) or not check_saved('kmeans_high_dim.pkl', run_path):
    print("KMeans models not found, calculate them first.")
    quit()

  kmeans = joblib.load(f'{run_path}/kmeans.pkl')
  kmeans_high_dim = joblib.load(f'{run_path}/kmeans_high_dim.pkl')

  N = 20000  # Adjust N if needed for quicker testing
  modes = ['normal', 'anomalous', 'random']
  if not check_saved (f'results_centroid_N_{N}.csv', run_path) or not check_saved (f'results_centroid_hd_N_{N}.csv', run_path) or not check_saved (f'results_kmeans_N_{N}.csv', run_path) or not check_saved (f'results_kmeans_hd_N_{N}.csv', run_path):
    print("Saved results not found, calculating new...")
    results_centroid = []
    results_centroid_hd = []
    results_kmeans = []
    results_kmeans_hd = []
  

    for MODE in modes:
        if check_saved(f'sequences_mode_{MODE}_N_{N}.pt', run_path):
            print(f"Loading sequences for mode {MODE}...")
            sequences = torch.load(f'{run_path}/sequences_mode_{MODE}_N_{N}.pt')
            column_names = torch.load(f'{run_path}/column_names_mode_{MODE}_N_{N}.pt')
        else:
            print(f"Sequences not found for mode {MODE}, calculating new...")
            sequences, column_names = get_sequences(csv_data, N, model_config['seq_len'], mode=MODE)
            torch.save(sequences, f'{run_path}/sequences_mode_{MODE}_N_{N}.pt')
            torch.save(column_names, f'{run_path}/column_names_mode_{MODE}_N_{N}.pt')

        for i, sequence in enumerate(tqdm.tqdm(sequences)):
            sequence_scaled = scale_sequence(scaler, sequence, device='cuda')
            _, embeddings, high_dim_embeddings = model(sequence_scaled)

            embeddings_np = embeddings.detach().cpu().numpy()
            high_dim_embeddings_np = high_dim_embeddings.detach().cpu().numpy()

            flattened_embeddings = embeddings_np.reshape(-1)  # This will flatten all dimensions
            flattened_high_dim_embeddings = high_dim_embeddings_np.reshape(-1)

            cosine_similarity = torch.nn.functional.cosine_similarity(embeddings, centroid, dim=1).mean().item()
            euclidean_distance = torch.nn.functional.pairwise_distance(embeddings, centroid, p=2).mean().item()
            cosine_similarity_high_dim = torch.nn.functional.cosine_similarity(high_dim_embeddings, centroid_high_dim, dim=1).mean().item()
            euclidean_distance_high_dim = torch.nn.functional.pairwise_distance(high_dim_embeddings, centroid_high_dim, p=2).mean().item()
            distance_closest_cluster = kmeans.transform(flattened_embeddings.reshape(1, -1)).min(axis=1).mean()
            distance_closest_cluster_high_dim = kmeans_high_dim.transform(flattened_high_dim_embeddings.reshape(1, -1)).min(axis=1).mean()
            results_centroid.append({'Mode': MODE, 'Cosine Similarity': cosine_similarity, 'Euclidean Distance': euclidean_distance})
            results_centroid_hd.append({'Mode': MODE, 'Cosine Similarity': cosine_similarity_high_dim, 'Euclidean Distance': euclidean_distance_high_dim})
            results_kmeans.append({'Mode': MODE, 'Distance to closest cluster': distance_closest_cluster})
            results_kmeans_hd.append({'Mode': MODE, 'Distance to closest cluster': distance_closest_cluster_high_dim})
  # Convert results to DataFrame
  
    results_centroid_df = pd.DataFrame(results_centroid)
    results_centroid_hd_df = pd.DataFrame(results_centroid_hd)
    results_kmeans_df = pd.DataFrame(results_kmeans)
    results_kmeans_hd_df = pd.DataFrame(results_kmeans_hd)
    
    # Save results
    results_centroid_df.to_csv(f'{run_path}/results_centroid_N_{N}.csv', index=False)
    results_centroid_hd_df.to_csv(f'{run_path}/results_centroid_hd_N_{N}.csv', index=False)
    results_kmeans_df.to_csv(f'{run_path}/results_kmeans_N_{N}.csv', index=False)
    results_kmeans_hd_df.to_csv(f'{run_path}/results_kmeans_hd_N_{N}.csv', index=False)
  else:
    print("Loading saved results...")
    results_centroid_df = pd.read_csv(f'{run_path}/results_centroid_N_{N}.csv')
    results_centroid_hd_df = pd.read_csv(f'{run_path}/results_centroid_hd_N_{N}.csv')
    results_kmeans_df = pd.read_csv(f'{run_path}/results_kmeans_N_{N}.csv')
    results_kmeans_hd_df = pd.read_csv(f'{run_path}/results_kmeans_hd_N_{N}.csv')

  hist_plot(data=results_kmeans_df, x='Distance to closest cluster', hue='Mode', element='step', bins=100, title='Distribution of Distance of embeddings to closest cluster Across Modes', xlabel='Distance to closest cluster', ylabel='Frequency')
  hist_plot(data=results_kmeans_hd_df, x='Distance to closest cluster', hue='Mode', element='step', bins=100, title='Distribution of Distance of HD embeddings to closest cluster Across Modes', xlabel='Distance to closest cluster', ylabel='Frequency')

  hist_plot(data=results_centroid_df, x='Cosine Similarity', hue='Mode', element='step', bins=100, title='Distribution of Cosine Similarity Across Modes', xlabel='Cosine Similarity', ylabel='Frequency')
  hist_plot(data=results_centroid_df, x='Euclidean Distance', hue='Mode', element='step', bins=100, title='Distribution of Euclidean Distance Across Modes', xlabel='Euclidean Distance', ylabel='Frequency')
  hist_plot(data=results_centroid_hd_df, x='Cosine Similarity', hue='Mode', element='step', bins=100, title='Distribution of Cosine Similarity Across Modes', xlabel='Cosine Similarity', ylabel='Frequency')
  hist_plot(data=results_centroid_hd_df, x='Euclidean Distance', hue='Mode', element='step', bins=100, title='Distribution of Euclidean Distance Across Modes', xlabel='Euclidean Distance', ylabel='Frequency')

  

