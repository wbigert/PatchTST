from ad_utils import init, inference, sample_with_ground_truth, check_saved
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import os
import json
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
def generate_dataset(run_path, data_path, flag, model_config, model):
    # if os.path.exists(os.path.join(run_path, f'{flag}_anomaly_detection_done.txt')):
    #     print(f"Anomaly detection dataset already generated for {flag}")
    #     return
    print(f"Generating {flag} dataset")
    seq_len = model_config['seq_len']
    pred_len = model_config['pred_len']
    data_scaled = pd.read_csv(run_path + '/data_scaled.csv')
    data_original = pd.read_csv(data_path)

    anomaly_sequences_path = run_path + f'/{flag}_anomaly_sequences.json'
    customer_map_path = run_path + f'/{flag}_customer_map.json'
    sms_anomaly_data_path = run_path + f'/{flag}_sms_anomaly.json'
    if flag =='train':
        indices = json.load(open(run_path + '/train_indices.json'))
    elif flag == 'val':
        indices = json.load(open(run_path + '/val_indices.json'))
    elif flag == 'test':
        indices = json.load(open(run_path + '/test_indices.json'))

    flat_indices = [idx for sequence in indices for idx in sequence]
    flat_indices.sort()
    unique_indices = list(dict.fromkeys(flat_indices))
    print(f"len unique_indices: {len(unique_indices)}")
    return unique_indices
    data_original = data_original.iloc[unique_indices]
    data_scaled = data_scaled.iloc[unique_indices]
    data_original.index = unique_indices
    data_scaled.index = unique_indices

    customer_ids = [int(customer_id) for customer_id in data_original['customer_id'].unique()]
    
    anomalous_ground_truth_map = {}

    if os.path.exists(customer_map_path):
        anomalous_ground_truth_map = json.load(open(customer_map_path))
        print("Loaded existing customer map")
    else:
        progress = tqdm(total=len(customer_ids), desc="Making customer_id maps", unit=" rows")
        for customer_id in customer_ids:
            other_indices = dict.fromkeys(data_original[data_original['customer_id'] != customer_id].index)
            other_ground_truths = [sequence[seq_len:] for sequence in indices if sequence[seq_len] in other_indices]
            anomalous_ground_truth_map[customer_id] = other_ground_truths
            progress.update(1)
        
        with open(os.path.join(customer_map_path), 'w') as f:
            json.dump(anomalous_ground_truth_map, f)



    if os.path.exists(anomaly_sequences_path):
        anomaly_detection_sequences = json.load(open(anomaly_sequences_path))
        print("Loaded existing anomaly_detection_sequences")
    else:
        progress = tqdm(total=len(indices), desc="Generating anomaly detection sequences", unit=" rows")
        anomaly_detection_sequences = []
        for i in range(len(indices)):
            input_rows = indices[i][:seq_len]
            ground_truth = indices[i][seq_len:]
            first_elem_idx = input_rows[0]
            first_elem = data_original.loc[first_elem_idx]
            customer_id = first_elem['customer_id']
            anomalous_options = anomalous_ground_truth_map[customer_id]
            rand_idx = np.random.randint(len(anomalous_options))
            anomalous_ground_truth = anomalous_options[rand_idx]
            progress.update(1)

            anomaly_detection_sequences.append({'input': input_rows, 'truth': ground_truth, 'anomaly': anomalous_ground_truth})
        
        with open(os.path.join(anomaly_sequences_path), 'w') as f:
            json.dump(anomaly_detection_sequences, f)

    if os.path.exists(sms_anomaly_data_path):
        sms_anomaly_data = json.load(open(sms_anomaly_data_path))
        print("Loaded existing sms_anomaly_data")
    else:
        sms_anomaly_data = []
        no_features = len(data_scaled.columns)
        progress = tqdm(total=len(anomaly_detection_sequences), desc="Computing anomaly detection dataset", unit=" rows")

        for data in anomaly_detection_sequences:
            input = torch.as_tensor(data_scaled.loc[data['input']].to_numpy(), dtype=torch.float32).to('cuda')
            input = input.unsqueeze(0)
            with torch.no_grad():
                output_batch, _, _ = model(input)
            output = output_batch.squeeze(0).cpu().numpy()
            anomaly_truth = data['anomaly']
            ground_truth = data['truth']
            feature_diffs_acc = {}
            feature_diffs_anomaly_acc = {}
            for feature_idx in range(no_features):
                feature_diffs_acc[feature_idx] = []
                feature_diffs_anomaly_acc[feature_idx] = []
            for i in range(pred_len):
                anomaly_truth_elem = data_scaled.loc[anomaly_truth[i]]
                ground_truth_elem = data_scaled.loc[ground_truth[i]]
                output_elem = output[i]
                
                for feature_idx in range(no_features):
                    feature_diffs_acc[feature_idx].append((output_elem[feature_idx] - ground_truth_elem[feature_idx])**2)
                    feature_diffs_anomaly_acc[feature_idx].append((output_elem[feature_idx] - anomaly_truth_elem[feature_idx])**2)
            
            feature_diffs = []
            feature_diffs_anomaly = []
            for i in range(no_features):
                feature_diffs.append(sum(feature_diffs_acc[i]) / pred_len)
                feature_diffs_anomaly.append(sum(feature_diffs_anomaly_acc[i]) / pred_len)

            sms_anomaly_data.append({'anomaly': 0, 'errors': feature_diffs})
            sms_anomaly_data.append({'anomaly': 1, 'errors': feature_diffs_anomaly})
            progress.update(1)


        with open(os.path.join(sms_anomaly_data_path), 'w') as f:
            json.dump(sms_anomaly_data, f)
      # write flag to txt to indicate finished
    with open(os.path.join(run_path, f'{flag}_anomaly_detection_done.txt'), 'w') as f:
        f.write('done')

def load_data(run_path, phase):
    sms_anomaly_data_path = f'{run_path}/{phase}_sms_anomaly.json'
    sms_anomaly_data = json.load(open(sms_anomaly_data_path))
    sms_anomaly_data = pd.DataFrame(sms_anomaly_data)
    
    X = np.array(sms_anomaly_data['errors'].to_list())
    y = np.array(sms_anomaly_data['anomaly'])
    return X, y

def train_anomaly_detector(run_path):
    X_train, y_train = load_data(run_path, 'train')
    models = {
        "Logistic Regression": LogisticRegression(random_state=0, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=0),
        "Gradient Boosting": GradientBoostingClassifier(random_state=0)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        # Optionally, save each model
        with open(os.path.join(run_path, f'{name.replace(" ", "_").lower()}_anomaly_detector.pkl'), 'wb') as f:
            pickle.dump(model, f)
    return trained_models

def evaluate_all_models(trained_models, run_path, phase):
    X, y = load_data(run_path, phase)
    for name, model in trained_models.items():
        print(f"Evaluating {name} on {phase} data...")
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"{name} {phase.capitalize()} Accuracy: {accuracy}")
        # precision, recall, f1-score with more decimal points
        print(classification_report(y, y_pred, digits=6))


if __name__ == '__main__':
    run_path ='./data/runs/model_PatchTST_name_sms_behavior_seqlen_100_predlen_1_epochs_35_patchlen_16_dmodel_128'
    data_path = 'D:/smsteknik-preprocess-test/aggregated/sms_behavior.csv'
    model, scaler, dataset_loader_args, model_config = init(run_path)

    unique_indices_train = generate_dataset(run_path, data_path, 'train', model_config, model)
    unique_indices_val = generate_dataset(run_path, data_path, 'val', model_config, model)
    unique_indices_test = generate_dataset(run_path, data_path, 'test', model_config, model)
    # ensure no overlap
    # assert len(set(unique_indices_train).intersection(unique_indices_val)) == 0, f"Overlap between train and val indices: {set(unique_indices_train).intersection(unique_indices_val)}"
    # assert len(set(unique_indices_train).intersection(unique_indices_test)) == 0, f"Overlap between train and test indices: {set(unique_indices_train).intersection(unique_indices_test)}"
    # assert len(set(unique_indices_val).intersection(unique_indices_test)) == 0, f"Overlap between val and test indices: {set(unique_indices_val).intersection(unique_indices_test)}"
    quit()

    trained_models = train_anomaly_detector(run_path)
    
    # Evaluate on validation data
    evaluate_all_models(trained_models, run_path, 'val')
    
    # Evaluate on test data
    evaluate_all_models(trained_models, run_path, 'test')
    