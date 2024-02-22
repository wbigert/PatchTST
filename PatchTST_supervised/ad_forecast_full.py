from ad_utils import init, inference, sample_with_ground_truth, check_saved
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
def run_forecast_anomaly_detection(run_path, n=30000, verbose=False):
    model, scaler, dataset_loader_args, model_config = init(run_path, transform=True)
    csv_data = pd.read_csv(dataset_loader_args['root_path'] + dataset_loader_args['data_path'])
    pred_len = model_config['pred_len']
    seq_len = model_config['seq_len']
    runs = [
        {"anomaly": True, "trivial": False},
        {"anomaly": False, "trivial": True},
        {"anomaly": False, "trivial": False},
        {"anomaly": True, "trivial": True}
    ]

    if check_saved('forecast_results.csv', run_path):
        print("Forecast results found, loading...")
        results_df = pd.read_csv(f'{run_path}/forecast_results.csv')
    else:
      results = []
      for run_type, run in enumerate(runs):
          ANOMALY = run["anomaly"]
          TRIVIAL = run["trivial"]

          for i in tqdm(range(n)):
              sequence, ground_truth, column_names_list = sample_with_ground_truth(csv_data, model_config['seq_len'], model_config['pred_len'], anomaly=ANOMALY)
              preds_scaled, preds_raw, trues_scaled, trues_raw = inference(model, scaler, sequence, pred_len, ground_truth, column_names_list, trivial=TRIVIAL, verbose=verbose)

              mse_scaled = np.mean((preds_scaled - trues_scaled) ** 2)
              prediction_name = 'trivial' if TRIVIAL else 'model'
              ground_truth = 'anomaly' if ANOMALY else 'actual'
              results.append({'Mode': f'ground_truth: {ground_truth}, prediction: {prediction_name}', 'Mean Squared Error': mse_scaled})
      results_df = pd.DataFrame(results)
      results_df.to_csv(f'{run_path}/forecast_results.csv', index=False)

    # Since log(0) is undefined, we need to make sure there are no zero MSE values
    # Adding a small constant (1e-9) to the Mean Squared Error to avoid log(0)
    results_df['Mean Squared Error'] += 1e-9

    # Apply a log transformation to the 'Mean Squared Error'
    results_df['Log Mean Squared Error'] = np.log(results_df['Mean Squared Error'])

    # Plotting the transformed data
    plt.figure(figsize=(12, 6))
    sns.histplot(data=results_df, x='Log Mean Squared Error', hue='Mode', element='step', bins=100)
    plt.title(f"Distribution of Log Mean Squared Error Across Modes. N: {n}, seq_len: {seq_len}, pred_len: {pred_len}.")
    plt.xlabel('Log Mean Squared Error')
    plt.ylabel('Frequency')
    plt.savefig(f'{run_path}/forecast_log_mse_distribution_N_{n}.png')

if __name__ == '__main__':
    run_forecast_anomaly_detection(run_path='./data/runs/model_PatchTST_name_weather_seqlen_100_predlen_40_epochs_50_patchlen_16_dmodel_128_dff_256', n=30000, verbose=False)