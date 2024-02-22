from ad_utils import init, inference, sample_with_ground_truth
import pandas as pd
import numpy as np

if __name__ == '__main__':
    run_path = './data/runs/model_PatchTST_name_weather_trimmed_seqlen_50_predlen_12_epochs_50_patchlen_16_dmodel_128_dff_256'
    model, scaler, dataset_loader_args, model_config = init(run_path, transform=True)
    csv_data = pd.read_csv(dataset_loader_args['root_path'] + dataset_loader_args['data_path'])
    pred_len = model_config['pred_len']
    seq_len = model_config['seq_len']
    enc_in = model_config['enc_in']
    # csv_data = pd.read_csv('./inference_input.csv')
    # csv_data = pd.read_csv('./test_input.csv', encoding='ISO-8859-1')
    VERBOSE = False
    ANOMALY = False
    TRIVIAL = True
    
    SHOW_OUTLIERS = False
    SHOW_FEATURE_WISE = False
    N = 10000

    preds_scaled_list = np.zeros((N, pred_len, enc_in))
    preds_raw_list = np.zeros((N, pred_len, enc_in))
    trues_scaled_list = np.zeros((N, pred_len, enc_in))
    trues_raw_list = np.zeros((N, pred_len, enc_in))

    for i in range(N):
      if i % 1000 == 0:
        print(f"Progress: {i} / {N}")
      sequence, ground_truth, column_names_list = sample_with_ground_truth(csv_data, seq_len, pred_len, anomaly=ANOMALY)
      preds_scaled, preds_raw, trues_scaled, trues_raw = inference(model, scaler, sequence, pred_len, ground_truth, column_names_list, trivial=TRIVIAL, verbose=VERBOSE)
      MSE = np.mean((preds_raw - trues_raw) ** 2)
      if MSE > 100000 and SHOW_OUTLIERS:
        print(f"MSE: {MSE}")
        for i, name in enumerate(column_names_list):
            diff = preds_raw[i] - trues_raw[i]
            print(f"Feature: {name}, diff={diff.mean().item()}, Prediction: {preds_raw[i].mean().item()}, Ground truth: {trues_raw[i].mean().item()}")
      preds_scaled_list[i] = preds_scaled
      preds_raw_list[i] = preds_raw
      trues_scaled_list[i] = trues_scaled
      trues_raw_list[i] = trues_raw
    print(f"Progress: {N} / {N}")
    if SHOW_OUTLIERS:
      mse_per_prediction = np.mean((preds_raw_list - trues_raw_list) ** 2, axis=(1, 2))
      worst_MSE_across_predictions = np.max(mse_per_prediction)
      mse_per_prediction_scaled = np.mean((preds_scaled_list - trues_scaled_list) ** 2, axis=(1, 2))
      worst_MSE_scaled_across_predictions = np.max(mse_per_prediction_scaled)
      print(f"worst_MSE_across_predictions: {worst_MSE_across_predictions}")
      print(f"worst_MSE_scaled_across_predictions: {worst_MSE_scaled_across_predictions}")

    mean_MSE_raw = np.mean((preds_raw_list - trues_raw_list) ** 2)
    mean_MSE_scaled = np.mean((preds_scaled_list - trues_scaled_list) ** 2)
    
    # Calculate the mean squared error per feature
    mse_per_feature = np.mean((preds_raw_list - trues_raw_list) ** 2, axis=(0, 1))
    mse_per_feature_scaled = np.mean((preds_scaled_list - trues_scaled_list) ** 2, axis=(0, 1))

    # Order by MSE_scaled
    if SHOW_FEATURE_WISE:
      permutation_scaled = np.argsort(mse_per_feature_scaled)
      column_names_list = np.array(column_names_list)
      column_names_ordered_by_scaled = column_names_list[permutation_scaled]
      mse_per_feature_ordered_by_scaled = mse_per_feature[permutation_scaled]
      mse_per_feature_scaled_ordered = mse_per_feature_scaled[permutation_scaled]

      # Print features ordered by MSE_scaled
      for i, name in enumerate(column_names_ordered_by_scaled):
          print(f"Feature: {name}, MSE: {mse_per_feature_ordered_by_scaled[i]}, MSE_scaled: {mse_per_feature_scaled_ordered[i]}")

    print(f"mean_MSE_raw: {mean_MSE_raw}")
    print(f"mean_MSE_scaled: {mean_MSE_scaled}")