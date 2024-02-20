from ad_utils import init, inference, take_random_sample
import pandas as pd
import numpy as np

if __name__ == '__main__':
    run_path = './data/runs/model_PatchTST_name_weather_seqlen_50_predlen_1_epochs_50_patchlen_16_dmodel_128_dff_256'
    model, scaler, dataset_loader_args, model_config = init(run_path)
    csv_data = pd.read_csv(dataset_loader_args['root_path'] + dataset_loader_args['data_path'])
    # csv_data = pd.read_csv('./inference_input.csv')
    # csv_data = pd.read_csv('./test_input.csv', encoding='ISO-8859-1')
    VERBOSE = False
    ANOMALY = False
    TRIVIAL = False
    N = 1000

    preds_scaled_list = np.zeros((N, model_config['pred_len'], model_config['enc_in']))
    preds_raw_list = np.zeros((N, model_config['pred_len'], model_config['enc_in']))
    trues_scaled_list = np.zeros((N, model_config['pred_len'], model_config['enc_in']))
    trues_raw_list = np.zeros((N, model_config['pred_len'], model_config['enc_in']))

    for i in range(N):
      if i % 1000 == 0:
        print(f"i: {i}")
      sequence, ground_truth, column_names_list = take_random_sample(csv_data, model_config['seq_len'], model_config['pred_len'], anomaly=ANOMALY)
      preds_scaled, preds_raw, trues_scaled, trues_raw = inference(model, scaler, sequence, ground_truth, column_names_list, trivial=TRIVIAL, verbose=VERBOSE)
      MSE = np.mean((preds_raw - trues_raw) ** 2)
      if MSE > 100000:
        print(f"MSE: {MSE}")
        # for each feature, print the difference between the prediction and the ground truth, along with the column name
        for i, name in enumerate(column_names_list):
            diff = preds_raw[i] - trues_raw[i]
            print(f"Feature: {name}, diff={diff.mean().item()}, Prediction: {preds_raw[i].mean().item()}, Ground truth: {trues_raw[i].mean().item()}")
      preds_scaled_list[i] = preds_scaled
      preds_raw_list[i] = preds_raw
      trues_scaled_list[i] = trues_scaled
      trues_raw_list[i] = trues_raw

    worst_MSE = np.max((preds_raw_list - trues_raw_list) ** 2)
    print(f"worst_MSE: {worst_MSE}")
    worst_MSE_scaled = np.max((preds_scaled_list - trues_scaled_list) ** 2)
    print(f"worst_MSE_scaled: {worst_MSE_scaled}")

    mean_MSE_raw = np.mean((preds_raw_list - trues_raw_list) ** 2)
    mean_MSE_scaled = np.mean((preds_scaled_list - trues_scaled_list) ** 2)

    print(f"mean_MSE_raw: {mean_MSE_raw}")
    print(f"mean_MSE_scaled: {mean_MSE_scaled}")