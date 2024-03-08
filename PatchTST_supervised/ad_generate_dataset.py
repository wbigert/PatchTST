from ad_utils import init, inference, sample_with_ground_truth, check_saved
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import os
import json
def generate_dataset(run_path, data_path):
    out_path = run_path + '/sms_anomaly.json'
    model, scaler, dataset_loader_args, model_config = init(run_path)
    seq_len = model_config['seq_len']
    pred_len = model_config['pred_len']
    data_scaled = pd.read_csv(run_path + '/data_scaled.csv')
    data_original = pd.read_csv(data_path)
    test_indices = json.load(open(run_path + '/test_indices.json'))
    train_indices = json.load(open(run_path + '/train_indices.json'))
    val_indices = json.load(open(run_path + '/val_indices.json'))

    print(f"type of train {type(train_indices)}")
    print(f"Length of {len(train_indices)}")
    flat_train_indices = [idx for sequence in train_indices for idx in sequence]
    unique_indices = list(dict.fromkeys(flat_train_indices))
    print(f"unique indices len: {len(unique_indices)}")
    print(f"len of data original: {len(data_original)}")
    data_original = data_original.iloc[unique_indices]

    customer_ids = data_original['customer_id'].unique()
    anomalous_ground_truth_map = {}
    progress = tqdm(total=len(customer_ids), desc="Making customer_id maps", unit=" rows")
    for customer_id in customer_ids:
        other_indices = dict.fromkeys(data_original[data_original['customer_id'] != customer_id].index)
        other_ground_truths = [sequence[seq_len:] for sequence in train_indices if sequence[seq_len] in other_indices]
        anomalous_ground_truth_map[customer_id] = other_ground_truths
        progress.update(1)

    progress = tqdm(total=len(train_indices), desc="Generating anomaly detection dataset", unit=" rows")
    anomaly_detection_data = []
    for i in range(len(train_indices)):
        input_rows = train_indices[i][:seq_len]
        ground_truth = train_indices[i][seq_len:]
        customer_id = int(data_original.iloc[input_rows[0]]['customer_id'])
        anomalous_ground_truth = np.random.choice(anomalous_ground_truth_map[customer_id])
        progress.update(1)

        anomaly_detection_data.append({'anomaly': 0, 'input': input_rows, 'truth': ground_truth})
        anomaly_detection_data.append({'anomaly': 1, 'input': input_rows, 'truth': anomalous_ground_truth})
    
    with open(os.path.join(out_path), 'w') as f:
        json.dump(anomaly_detection_data, f)

if __name__ == '__main__':
    run_path ='./data/runs/model_PatchTST_name_sms_behavior_seqlen_100_predlen_1_epochs_35_patchlen_16_dmodel_128'
    data_path = 'D:/smsteknik-preprocess-test/aggregated/sms_behavior.csv'
    generate_dataset(run_path, data_path)