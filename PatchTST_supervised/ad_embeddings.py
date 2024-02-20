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