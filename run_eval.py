from Models.LSDataset_eval import LSDataset
from Models.tcn_tsa_eval import TCN_Temporal_Attention_Landslide

import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

import os
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.utils.data import DataLoader
import datetime
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    config = {'data_path': './data/landslide_hourly_antecedent_lai_only_rainfall_new_rainfallwithin_96_hours/', 'max_et_length':15, 'input_lai': True,
              'input_geoenv': True, 'input_size_event': 15*24, 'hidden_dim': 16, 'output_dim': 32, 'embed_dim': 64, 'lai_embed_dim': 64,'input_size_ante': 30,
              'geo_env_hidden_dim': 64, 'geo_env_dim': 8, 'static_vegetation': False, 'test_sequence_len':'all', 'num_class':1, 'return_attention': True, 'n_head': 8, 'dropout':0.1}


    All_Results = ['Datasets', 'ConvTran']  # Use to store the accuracy of ConvTran in e.g "Result/Datasets/UEA"
    datafile_name = 'landslide_database.csv'
    anterainfall_filename = 'ante_rainfall.csv'
    ante_et_filename = 'ante_et_2.csv'
    ante_lai_filename = 'ante_lai.csv'
    et_mask_filename = 'et_masks.csv'

    wind_speed_filename = 'wind_speed.csv'
    temperature_filename = 'Temperature.csv'
    pressure_filename = 'Pressure.csv'
    sun_radiance_filename = 'Sun_radiance.csv'
    relative_humidity_filename = 'RelativeHumidity.csv'



    rainfall_event_dir = 'rainfall_event'
    # rainfall_event_dir = 'last_rainfall_event'
    geo_env_filename = 'geo_env.csv'


    test_dataset = LSDataset(data_path=os.path.join(config['data_path'], 'test', datafile_name),
                              rainfall_event_path=os.path.join(config['data_path'], 'test', rainfall_event_dir),
                              ante_lai_path=os.path.join(config['data_path'], 'test', ante_lai_filename),
                              ante_et_mask_path=os.path.join(config['data_path'], 'test', et_mask_filename),
                              wind_path=os.path.join(config['data_path'], 'test', wind_speed_filename),
                              temperature_path=os.path.join(config['data_path'], 'test', temperature_filename),
                              pressure_path=os.path.join(config['data_path'], 'test', pressure_filename),
                              sun_radiance_path=os.path.join(config['data_path'], 'test', sun_radiance_filename),
                              relative_humidity_path=os.path.join(config['data_path'], 'test',
                                                                 relative_humidity_filename),
                              maximum_length=config['input_size_event'],
                              max_et_length=config['max_et_length'],
                              padding_value=0,
                              geo_env_filename=os.path.join(config['data_path'], 'test',geo_env_filename),
                              static_vegetation=config['static_vegetation'],
                              rainfall_sequence_length=config['test_sequence_len'])

    test_loader = DataLoader(dataset=test_dataset, batch_size=24, shuffle=True, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config['static_vegetation']:
        config['geo_env_dim'] = 10
        config['input_et'] = False

    model = TCN_Temporal_Attention_Landslide(config=config)
    model.to(device)
    model.load_state_dict(
        torch.load('./lai_geoenv_test_all_model_all.pth'))
    model.to(device)
    model.eval()
    predict_list = []
    target_list = []
    IDs_list = []
    lai_features_list = []
    prob_list = []



    with torch.no_grad():
        for batch_idx, (rainfall_event, lai_data, rainfall_mask, et_masks, targets, geo_env, IDs) in enumerate(
                test_loader):
            # outputs = model(geo_env)
            outputs, features = model(rainfall_event.to(device), geo_env.to(device), lai_data.to(device),
                                  et_masks.to(device), rainfall_mask.to(device))
            attn = features[0]
            attn_arr = attn.cpu().numpy()
            lai_features_list.append(features[1].detach().cpu().numpy())
            predict = (outputs > 0.5).long()
            prob_list.append(outputs.squeeze(dim=1).detach().cpu().numpy())
            predict_list.append(predict.squeeze(dim=1).detach().cpu().numpy())
            target_list.append(targets.cpu().detach().numpy())
            IDs_list.extend(IDs)
    predict_arr = np.concatenate(predict_list, axis=0)
    prob_arr = np.concatenate(prob_list, axis=0)
    target_arr = np.concatenate(target_list, axis=0)

    results = {'y_pred': predict_arr, 'y_target': target_arr, 'INCID': IDs_list}
    results_all_df = pd.DataFrame(results)

    predict_arr = np.concatenate(predict_list, axis=0)
    target_arr = np.concatenate(target_list, axis=0)
    # lai_features = np.concatenate(lai_features_list, axis=0)
    accuracy_test = accuracy_score(target_arr, predict_arr)
    f1_test = f1_score(target_arr, predict_arr)
    precision_test = precision_score(target_arr, predict_arr)
    recall_test = recall_score(target_arr, predict_arr)
    print(
        f'accuracy: {accuracy_test}, precision: {precision_test}, recall:{recall_test}, f1_score:{f1_test}')

