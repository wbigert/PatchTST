import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from ad_utils import init, get_sequences, scale_sequence, check_saved, hist_plot
import joblib
import tqdm

def evaluate_embeddings(run_path, n=20000, k_clusters=10):
  model, scaler, dataset_loader_args, model_config = init(run_path, transform=True)
  
  if not check_saved('centroid_high_dim.npy', run_path) or not check_saved('centroid.npy', run_path):
    print("Centroid not found, calculate them first.")
    quit()
    
  centroid = torch.as_tensor(np.load(f'{run_path}/centroid.npy'), dtype=torch.float32).to('cuda')
  centroid_high_dim = torch.as_tensor(np.load(f'{run_path}/centroid_high_dim.npy'), dtype=torch.float32).to('cuda')
  csv_data = pd.read_csv(dataset_loader_args['root_path'] + dataset_loader_args['data_path'])
  
  if not check_saved(f'kmeans_k_{k_clusters}.pkl', run_path) or not check_saved(f'kmeans_high_dim_k_{k_clusters}.pkl', run_path):
    print(f"KMeans models not found for {k_clusters} clusters, calculate them first.")
    quit()

  kmeans = joblib.load(f'{run_path}/kmeans_k_{k_clusters}.pkl')
  kmeans_high_dim = joblib.load(f'{run_path}/kmeans_high_dim_k_{k_clusters}.pkl')

  modes = ['normal', 'anomalous', 'random']
  if not check_saved (f'results_centroid_N_{n}.csv', run_path) or not check_saved (f'results_centroid_hd_N_{n}.csv', run_path) or not check_saved (f'results_kmeans_N_{n}_k_{k_clusters}.csv', run_path) or not check_saved (f'results_kmeans_hd_N_{n}_k_{k_clusters}.csv', run_path):
    print("Saved results not found, calculating new...")
    results_centroid = []
    results_centroid_hd = []
    results_kmeans = []
    results_kmeans_hd = []
  

    for MODE in modes:
        if check_saved(f'sequences_mode_{MODE}_N_{n}.pt', run_path):
            print(f"Loading sequences for mode {MODE}...")
            sequences = torch.load(f'{run_path}/sequences_mode_{MODE}_N_{n}.pt')
            column_names = torch.load(f'{run_path}/column_names_mode_{MODE}_N_{n}.pt')
        else:
            print(f"Sequences not found for mode {MODE}, calculating new...")
            sequences, column_names = get_sequences(csv_data, n, model_config['seq_len'], mode=MODE)
            torch.save(sequences, f'{run_path}/sequences_mode_{MODE}_N_{n}.pt')
            torch.save(column_names, f'{run_path}/column_names_mode_{MODE}_N_{n}.pt')

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
    results_centroid_df.to_csv(f'{run_path}/results_centroid_N_{n}.csv', index=False)
    results_centroid_hd_df.to_csv(f'{run_path}/results_centroid_hd_N_{n}.csv', index=False)
    results_kmeans_df.to_csv(f'{run_path}/results_kmeans_N_{n}_k_{k_clusters}.csv', index=False)
    results_kmeans_hd_df.to_csv(f'{run_path}/results_kmeans_hd_N_{n}_k_{k_clusters}.csv', index=False)
  else:
    print("Loading saved results...")
    results_centroid_df = pd.read_csv(f'{run_path}/results_centroid_N_{n}.csv')
    results_centroid_hd_df = pd.read_csv(f'{run_path}/results_centroid_hd_N_{n}.csv')
    results_kmeans_df = pd.read_csv(f'{run_path}/results_kmeans_N_{n}_k_{k_clusters}.csv')
    results_kmeans_hd_df = pd.read_csv(f'{run_path}/results_kmeans_hd_N_{n}_k_{k_clusters}.csv')

  hist_plot(save_path=run_path + f'/hist_kmeans_N_{n}_k_{k_clusters}.png', data=results_kmeans_df, x='Distance to closest cluster', hue='Mode', element='step', bins=100, title='Distance of Embeddings to closest cluster Across Modes', xlabel='Distance to closest cluster', ylabel='Frequency')
  hist_plot(save_path=run_path + f'/hist_kmeans_hd_N_{n}_k_{k_clusters}.png', data=results_kmeans_hd_df, x='Distance to closest cluster', hue='Mode', element='step', bins=100, title='Distance of HD Embeddings to closest cluster Across Modes', xlabel='Distance to closest cluster', ylabel='Frequency')
  hist_plot(save_path=run_path + f'/cosine_N_{n}.png', data=results_centroid_df, x='Cosine Similarity', hue='Mode', element='step', bins=100, title='Cosine Similarity of Embeddings Across Modes', xlabel='Cosine Similarity', ylabel='Frequency')
  hist_plot(save_path=run_path + f'/euc_N_{n}.png', data=results_centroid_df, x='Euclidean Distance', hue='Mode', element='step', bins=100, title='Euclidean Distance of Embeddings Across Modes', xlabel='Euclidean Distance', ylabel='Frequency')
  hist_plot(save_path=run_path + f'/cosine_hd_N_{n}.png', data=results_centroid_hd_df, x='Cosine Similarity', hue='Mode', element='step', bins=100, title='Cosine Similarity of HD Embeddings Across Modes', xlabel='Cosine Similarity', ylabel='Frequency')
  hist_plot(save_path=run_path + f'/euc_hd_N_{n}.png', data=results_centroid_hd_df, x='Euclidean Distance', hue='Mode', element='step', bins=100, title='Euclidean Distance of HD Embeddings Across Modes', xlabel='Euclidean Distance', ylabel='Frequency')

if __name__ == '__main__':
  evaluate_embeddings('./data/runs/model_PatchTST_name_weather_seqlen_50_predlen_1_epochs_50_patchlen_16_dmodel_128_dff_256', n=20000, k_clusters=10)

