import os
import pandas as pd
import json
from ad_utils import run_custom
from ad_forecast_full import run_forecast_anomaly_detection
from ad_calc_embedding_centroid import process_embeddings
from ad_embeddings import evaluate_embeddings

if __name__ == '__main__':
  for pred_len in [1]: # [5, 10]
    for patch_len in [8, 16]:
      for d_model in [16, 32, 48, 64, 80, 96, 112, 128, 144, 160]:
        print(f"Running for d_model={d_model} with patch_len={patch_len}")
        run_path = run_custom(d_model=d_model, d_ff=d_model*2, train_epochs=35, seq_len=50, pred_len=pred_len, patch_len=patch_len, model='PatchTST', data_name='weather')
        print(f"Completed run for d_model={d_model} with run_path: {run_path}")
        if os.path.getsize(f'{run_path}/stderr.log') == 0:
          print(f"SUCCESS: {run_path}")
          print(f"Running forecast anomaly detection for {run_path}")
          run_forecast_anomaly_detection(run_path, n=20000, verbose=False)
          print(f"Completed forecast anomaly detection for {run_path}")
          print(f"Running process_embeddings for {run_path}")
          process_embeddings(run_path, explore_centroids=False, k_clusters=10)
          print(f"Completed process_embeddings for {run_path}")
          print(f"Running evaluate_embeddings for {run_path}")
          evaluate_embeddings(run_path, n=20000, k_clusters=10)
          print(f"Completed evaluate_embeddings for {run_path}")
        else:
          print(f"ERROR: {run_path}")
  print("All runs completed")
    
