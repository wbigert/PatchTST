
import json
from models import PatchTST
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_Pred
import torch
import joblib

class Configs:
      def __init__(self, **kwargs):
          for k, v in kwargs.items():
              setattr(self, k, v)

def get_model(run_path, model_config):
    configs = Configs(**model_config)
    model = PatchTST.Model(configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(run_path + '/checkpoint.pth', map_location=device))
    model.double()
    model = model.to(device)
    model.eval()
    return model

def get_scaler(run_path):
    scaler_path = run_path + '/scaler.joblib'
    scaler = joblib.load(scaler_path)
    return scaler

def get_data_loader(dataset_loader_args, input_data_path, scaler):
  dataset_loader_args['data_path'] = input_data_path
  dataset_loader_args['scaler'] = scaler
  args = Configs(**dataset_loader_args)
  dataset_pred = Dataset_Pred(args)
  data_loader = DataLoader(dataset_pred, batch_size=1, shuffle=False, drop_last=False)
  return data_loader


def inference(run_path, input_data_path):
    # load dataset_loader_args.json
    with open(run_path + '/dataset_loader_args.json') as f:
        dataset_loader_args = json.load(f)
    
    # load model_config.json
    with open(run_path + '/model_config.json') as f:
        model_config = json.load(f)
    
    model = get_model(run_path, model_config)
    scaler = get_scaler(run_path)
    data_loader = get_data_loader(dataset_loader_args, input_data_path, scaler)

    for seq_x, _, _, _ in data_loader:
        seq_x = seq_x.double().to(model.device)
        with torch.no_grad():
            outputs = model(seq_x)
            print(outputs)
    