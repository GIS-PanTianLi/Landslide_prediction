import torch
from torch import nn
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
import pickle

def rolling_window(a, window):
    shape = a.shape[-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



def getMRR(x, path ,window_size):
    x = x.replace('/', '_')
    filename = path + x + '.npy'
    rainfall_event = np.load(filename, allow_pickle=True)
    if window_size > len(rainfall_event):
        mrr_24 = 0
    else:
        rolling_rainfall_array = sliding_window_view(rainfall_event, window_size)
        rolling_rainfall_sum = rolling_rainfall_array.sum(axis=1)
        mrr_24 = rolling_rainfall_sum.max()
    return mrr_24


def getMRR_event(rainfall_event ,window_size):
    if window_size > len(rainfall_event):
        mrr_24 = 0
    else:
        rolling_rainfall_array = sliding_window_view(rainfall_event, window_size)
        rolling_rainfall_sum = rolling_rainfall_array.sum(axis=1)
        mrr_24 = rolling_rainfall_sum.max()
    return mrr_24

def getAntecedentRainfall_event(rainfall_event, days):

    start_index = len(rainfall_event)-days*24
    at = rainfall_event[start_index:]
    cummulative_at = at.sum()
    return cummulative_at


def getAntecedentRainfall(x, path , days):
    x = x.replace('/', '_')
    filename = path + x + '.npy'
    rainfall_event = np.load(filename, allow_pickle=True)
    start_index = len(rainfall_event)-days*24
    at = rainfall_event[start_index:]
    cummulative_at = at.sum()
    return cummulative_at


def getX_Y(geoenv_path, database_path, rainfall_event_path, last_rainfall_path,ante_rainfall_path=None, et_path=None,wind_path=None,
       temperature_path=None,pressure_path=None,
       sun_radiance_path=None, relative_humidity_path=None):

    static_features = pd.read_csv(geoenv_path)
    geo_type_dict = {'Fill': 1, 'Superficial deposits': 2, 'Volcanic rocks': 3, 'Sedimentary rocks': 4,
                     'Intrusive rocks': 5}
    static_features['Coarse_G'] = static_features.replace({'Coarse_G': geo_type_dict})['Coarse_G']
    x_value = static_features[['INCID', 'water_dist', 'fault_dist', 'elev', 'slope',
           'pro_curva', 'plan_curva', 'aspect', 'Coarse_G']]
    ante_et = pd.read_csv(et_path)
    wind_speed = pd.read_csv(wind_path)
    temperature = pd.read_csv(temperature_path)
    pressure = pd.read_csv(pressure_path)
    sun_radiance = pd.read_csv(sun_radiance_path)
    relative_humidity = pd.read_csv(relative_humidity_path)

    # ante_rainfall = pd.read_csv(ante_rainfall_path)
    x_value['AR_1d'] = x_value['INCID'].apply(getAntecedentRainfall, path=rainfall_event_path, days=1)
    x_value['AR_3d'] = x_value['INCID'].apply(getAntecedentRainfall, path=rainfall_event_path, days=3)
    x_value['AR_7d'] = x_value['INCID'].apply(getAntecedentRainfall, path=rainfall_event_path, days=7)
    x_value['AR_15d'] = x_value['INCID'].apply(getAntecedentRainfall, path=rainfall_event_path, days=15)

    x_value['lai_mean'] = ante_et.iloc[:,1:16].mean(axis=1)
    x_value['wind_mean'] = wind_speed.iloc[:,2:17].mean(axis=1)
    x_value['temperature_mean'] = temperature.iloc[:,2:17].mean(axis=1)
    x_value['pressure_mean'] = pressure.iloc[:,2:17].mean(axis=1)
    x_value['sun_radiance_mean'] = sun_radiance.iloc[:,2:17].mean(axis=1)
    x_value['relative_humidity_mean'] = relative_humidity.iloc[:,2:17].mean(axis=1)



    x_value['24_mrr'] = x_value['INCID'].apply(getMRR, path=last_rainfall_path, window_size=24)
    x_value['12_mrr'] = x_value['INCID'].apply(getMRR, path=last_rainfall_path, window_size=12)
    x_value['4_mrr'] = x_value['INCID'].apply(getMRR, path=last_rainfall_path, window_size=4)
    x_value['1_mrr'] = x_value['INCID'].apply(getMRR, path=last_rainfall_path, window_size=1)

    train_landslide = pd.read_csv(database_path)
    y_value = train_landslide['label'].astype(int)
    x_value['label'] = y_value
    x_value['Rainfall_events'] = train_landslide['Rainfall_events']

    return x_value[['water_dist', 'fault_dist', 'elev', 'slope',
           'pro_curva', 'plan_curva', 'aspect', 'Coarse_G', 'AR_1d',
            'AR_3d', 'AR_7d', 'AR_15d', 'lai_mean', 'wind_mean','temperature_mean','pressure_mean','sun_radiance_mean','relative_humidity_mean', '1_mrr', '4_mrr', '12_mrr', '24_mrr', 'INCID', 'label', 'Rainfall_events']], x_value['label']



class CNN_LSTM(nn.Module):
    def __init__(self, in_channels):
        super(CNN_LSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=1,stride=1),
            nn.ReLU(),
            nn.MaxPool1d(1, 2),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(1, 2),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(1, 2),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(1, 2),
            nn.Dropout(0.1)
        )
        self.ffn1 = nn.Sequential(
            nn.Linear(512, 200),
            nn.Linear(200, 20),
            nn.Linear(20, 2),
            nn.Softmax(dim=2)
        )
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.ffn2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # x = self.ffn1(x)
        x = x.squeeze(dim=1)
        # x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.lstm_dropout(x)
        x, _ = self.lstm2(x)
        x = self.lstm_dropout(x)
        x = self.ffn2(x)

        return x


class dnn_model(nn.Module):
    def __init__(self, in_channels):
        super(dnn_model, self).__init__()
        self.hidden=nn.Sequential(
            nn.Linear(in_channels,30),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(30, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(30,10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            # nn.Dropout(0.1),
        )
        self.classfication=nn.Sequential(
            nn.Linear(10,2),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = x.squeeze(dim=1)
        x=self.hidden(x)
        x=self.classfication(x)
        return x


if __name__ == '__main__':

    test_x, test_y = getX_Y(
        geoenv_path='./data/landslide_hourly_antecedent_lai_only_rainfall_new_rainfallwithin_96_hours/test/geo_env.csv',
        ante_rainfall_path='./data/landslide_hourly_antecedent_lai_only_rainfall_new_rainfallwithin_96_hours/test/ante_rainfall.csv',
        database_path='./data/landslide_hourly_antecedent_lai_only_rainfall_new_rainfallwithin_96_hours/test/landslide_database.csv',
        rainfall_event_path='./data/landslide_hourly_antecedent_lai_only_rainfall_new_rainfallwithin_96_hours/test/rainfall_event/',
        last_rainfall_path='./data/landslide_hourly_antecedent_lai_only_rainfall_new_rainfallwithin_96_hours/test/last_rainfall_event/',
        et_path='./data/landslide_hourly_antecedent_lai_only_rainfall_new_rainfallwithin_96_hours/test/ante_lai.csv',
        wind_path='./data//landslide_hourly_antecedent_lai_only_rainfall_new_rainfallwithin_96_hours/test/wind_speed.csv',
        temperature_path='./data//landslide_hourly_antecedent_lai_only_rainfall_new_rainfallwithin_96_hours/test/Temperature.csv',
        pressure_path='./data/landslide_hourly_antecedent_lai_only_rainfall_new_rainfallwithin_96_hours/test/Pressure.csv',
        sun_radiance_path='./data/landslide_hourly_antecedent_lai_only_rainfall_new_rainfallwithin_96_hours/test/Sun_radiance.csv',
        relative_humidity_path='./data/landslide_hourly_antecedent_lai_only_rainfall_new_rainfallwithin_96_hours/test/RelativeHumidity.csv'
    )

    feature_columns = ['1_mrr', '4_mrr', '12_mrr', '24_mrr', 'AR_1d',
                       'AR_3d', 'AR_7d', 'AR_15d'] + ['aspect', 'Coarse_G', 'elev', 'fault_dist', 'plan_curva', 'pro_curva',
                                                      'slope', 'water_dist'] + ['wind_mean', 'temperature_mean',
                                                                                'pressure_mean', 'sun_radiance_mean',
                                                                                'relative_humidity_mean', 'lai_mean']
    # transfer = StandardScaler()
    with open('scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    x_test = test_x[feature_columns].values
    x_test = loaded_scaler.transform(x_test)

    x_test, y_test = torch.from_numpy(x_test.astype(np.float32)), torch.from_numpy(
        test_y.astype(np.int64).values.reshape(-1, 1))
    x_test = x_test.unsqueeze(dim=1)
    test_data = Data.TensorDataset(x_test, y_test)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=24, shuffle=True, pin_memory=True)
    model = CNN_LSTM(in_channels=22)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(
        torch.load('./cnn_lstm_best_model_paper.pth'))

    model.eval()
    correct = 0
    total = 0
    predict_list = []
    target_list = []
    IDs_list = []
    lai_features_list = []
    with torch.no_grad():
        for step, (b_x, b_y) in enumerate(test_loader):
            outputs = model(b_x.to(device))
            outputs = outputs.argmax(dim=1, keepdim=True)
            # predict = (outputs > 0.5).long()
            predict_list.append(outputs.squeeze(dim=1).detach().cpu().numpy())
            target_list.append(b_y.cpu().detach().numpy())

        predict_arr = np.concatenate(predict_list, axis=0)
        target_arr = np.concatenate(target_list, axis=0)
        # lai_features = np.concatenate(lai_features_list, axis=0)
        accuracy_test = accuracy_score(target_arr, predict_arr)
        f1_test = f1_score(target_arr, predict_arr)
        precision_test = precision_score(target_arr, predict_arr)
        recall_test = recall_score(target_arr, predict_arr)
        print(
            f'accuracy: {accuracy_test}, precision: {precision_test}, recall:{recall_test}, f1_score:{f1_test}')