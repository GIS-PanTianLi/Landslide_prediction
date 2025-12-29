import pandas as pd
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
import os
from typing import Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import torch

class LSDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 rainfall_event_path: str,
                 maximum_length: int,
                 max_et_length: int,
                 padding_value: int,
                 geo_env_filename: str,
                 ante_lai_path: Optional[str] = None,
                 ante_et_mask_path: Optional[str] = None,
                 wind_path: Optional[str] = None,
                 temperature_path: Optional[str] = None,
                 pressure_path: Optional[str] = None,
                 sun_radiance_path: Optional[str] = None,
                 relative_humidity_path : Optional[str] = None,
                 ante_et_path: Optional[str] = None,
                 static_vegetation: Optional[str] = None,
                 rainfall_sequence_length: Optional[int] = None,

                 **kwargs):
        super().__init__(**kwargs)

        self.data_path = data_path
        self.rainfall_event_path = rainfall_event_path
        self.ante_lai_path = ante_lai_path
        self.ante_et_mask_path = ante_et_mask_path
        self.wind_speed_path = wind_path
        self.temperature_path = temperature_path
        self.pressure_path = pressure_path
        self.sun_radiance_path = sun_radiance_path
        self.relative_humidity_path = relative_humidity_path
        self.maximum_length = maximum_length
        self.max_et_length = max_et_length
        self.geo_env_filename = geo_env_filename
        self.static_vegetation = static_vegetation
        if rainfall_sequence_length:
            self.rainfall_sequence_length = rainfall_sequence_length
        else:
            self.rainfall_sequence_length = None
        self._load_points()
        self.rainfall_event_max = 170.77885818873253
        self.padding_value = padding_value


    def getWeight(self):
        labels = self.dataset['label']
        weights = [1 / (len(list(filter(lambda x: x == i, labels))) + 1e-5) for i in labels]
        return weights


    def _load_points(self):

        def getRainfallEventSequenceLength(x):
            filename = x['INCID']
            path = os.path.join(self.rainfall_event_path,filename + '.npy')
            # path = os.path.join(self.rainfall_event_path, filename + '_last_event.npy')
            data = np.load(path, allow_pickle=True)
            data_len = len(data)
            return data_len

        # load data
        self.dataset = pd.read_csv(self.data_path)

        self.dataset['INCID'] = self.dataset['INCID'].str.replace('/', '_')
        self.dataset['Rainfall_length'] = self.dataset.apply(getRainfallEventSequenceLength, axis=1)

        minmax_scaler = MinMaxScaler()  # 建立MinMaxScaler模型对象


        if self.ante_lai_path != None:
            self.ante_lai = pd.read_csv(self.ante_lai_path)
            self.ante_lai['INCID'] = self.ante_lai['INCID'].str.replace('/', '_')
            self.ante_lai['Rainfall_length'] = self.ante_lai.apply(getRainfallEventSequenceLength, axis=1)
            ante_lai_columns = self.ante_lai.columns[self.ante_lai.columns.str.contains('lai_pre')].tolist()
            self.ante_lai[ante_lai_columns] = minmax_scaler.fit_transform(self.ante_lai[ante_lai_columns])  # MinMaxScaler标准化处理
            self.et_mask = pd.read_csv(self.ante_et_mask_path)

        if self.wind_speed_path != None:
            self.wind_speed = pd.read_csv(self.wind_speed_path)
            self.wind_speed['INCID'] = self.wind_speed['INCID'].str.replace('/', '_')
            self.wind_speed['Rainfall_length'] = self.wind_speed.apply(getRainfallEventSequenceLength, axis=1)
            wind_speed_columns = self.wind_speed.columns[self.wind_speed.columns.str.contains('ws_Pre_')].tolist()
            self.wind_speed[wind_speed_columns] = minmax_scaler.fit_transform(self.wind_speed[wind_speed_columns])  # MinMaxScaler标准化处理

        if self.temperature_path != None:
            self.temperature = pd.read_csv(self.temperature_path)
            self.temperature['INCID'] = self.temperature['INCID'].str.replace('/', '_')
            self.temperature['Rainfall_length'] = self.temperature.apply(getRainfallEventSequenceLength, axis=1)
            temperature_columns = self.temperature.columns[self.temperature.columns.str.contains('T_Pre_')].tolist()
            self.temperature[temperature_columns] = minmax_scaler.fit_transform(self.temperature[temperature_columns])  # MinMaxScaler标准化处理

        if self.pressure_path != None:
            self.pressure = pd.read_csv(self.pressure_path)
            self.pressure['INCID'] = self.pressure['INCID'].str.replace('/', '_')
            self.pressure['Rainfall_length'] = self.pressure.apply(getRainfallEventSequenceLength, axis=1)
            pressure_columns = self.pressure.columns[self.pressure.columns.str.contains('P_Pre_')].tolist()
            self.pressure[pressure_columns] = minmax_scaler.fit_transform(self.pressure[pressure_columns])  # MinMaxScaler标准化处理

        if self.sun_radiance_path != None:
            self.sun_radiance = pd.read_csv(self.sun_radiance_path)
            self.sun_radiance['INCID'] = self.sun_radiance['INCID'].str.replace('/', '_')
            self.sun_radiance['Rainfall_length'] = self.sun_radiance.apply(getRainfallEventSequenceLength, axis=1)
            sun_radiance_columns = self.sun_radiance.columns[self.sun_radiance.columns.str.contains('sun_Pre_')].tolist()
            self.sun_radiance[sun_radiance_columns] = minmax_scaler.fit_transform(self.sun_radiance[sun_radiance_columns])  # MinMaxScaler标准化处理

        if self.relative_humidity_path != None:
            self.relative_humidity = pd.read_csv(self.relative_humidity_path)
            self.relative_humidity['INCID'] = self.relative_humidity['INCID'].str.replace('/', '_')
            self.relative_humidity['Rainfall_length'] = self.relative_humidity.apply(getRainfallEventSequenceLength, axis=1)
            relative_humidity_columns = self.relative_humidity.columns[self.relative_humidity.columns.str.contains('Rh_Pre_')].tolist()
            self.relative_humidity[relative_humidity_columns] = minmax_scaler.fit_transform(self.relative_humidity[relative_humidity_columns])  # MinMaxScaler标准化处理


        self.geo_env = pd.read_csv(self.geo_env_filename)
        self.geo_env['INCID'] = self.geo_env['INCID'].str.replace('/', '_')
        self.geo_env['Rainfall_length'] = self.geo_env.apply(getRainfallEventSequenceLength, axis=1)

        if self.static_vegetation:
            geo_env_columns = ['water_dist', 'fault_dist', 'elev', 'slope','pro_curva', 'plan_curva', 'aspect', 'lai_mean']
        else:
            geo_env_columns = ['water_dist', 'fault_dist', 'elev', 'slope','pro_curva', 'plan_curva', 'aspect']
        # self.geo_env[geo_env_columns] = (self.geo_env[geo_env_columns] - self.geo_env[geo_env_columns].min()) / (self.geo_env[geo_env_columns].max() - self.geo_env[geo_env_columns].min())
        geo_type_dict = {'Fill':1, 'Superficial deposits': 2, 'Volcanic rocks': 3, 'Sedimentary rocks':4, 'Intrusive rocks': 5}
        self.geo_env['Coarse_G'] = self.geo_env.replace({'Coarse_G': geo_type_dict})['Coarse_G']
        transfer = StandardScaler()
        self.geo_env[geo_env_columns] = transfer.fit_transform(self.geo_env[geo_env_columns].values)
        if self.rainfall_sequence_length != 'all':
            self.dataset = self.dataset[self.dataset['Rainfall_length'] >= self.rainfall_sequence_length]
            self.dataset.index= np.arange(self.dataset.shape[0])
            self.ante_rainfall = self.ante_rainfall[self.ante_rainfall['Rainfall_length'] >= self.rainfall_sequence_length]
            self.ante_rainfall.index = np.arange(self.ante_rainfall.shape[0])
            self.geo_env = self.geo_env[self.geo_env['Rainfall_length'] >= self.rainfall_sequence_length]
            self.geo_env.index = np.arange(self.geo_env.shape[0])
        self.data_len = self.dataset.shape[0]

    def create_padding_mask(self, x, pad_token=-999):
        mask = (x != pad_token)
        return mask

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        incid_id = self.dataset.loc[idx, 'INCID']
        incid_id = incid_id.replace('/','_')
        # rainfall_event_path = self.rainfall_event_path + f'/{incid_id}_last_event.npy'
        rainfall_event_path = self.rainfall_event_path + f'/{incid_id}.npy'
        if self.rainfall_sequence_length == 'all':
            rainfall_event = torch.from_numpy(np.load(rainfall_event_path, allow_pickle=True).astype(np.float32))

        else:
            rainfall_event = torch.from_numpy(np.load(rainfall_event_path)[0:self.rainfall_sequence_length].astype(np.float32))
        if self.ante_lai_path != None:
            ante_lai_columns = ['lai_pre' + str(x) for x in range(self.max_et_length-1,-1,-1)]
            # ante_et = torch.from_numpy(self.ante_et.iloc[idx, 1:31].values.astype(np.float32))
            lai_data = torch.from_numpy(self.ante_lai.loc[idx, ante_lai_columns].values.astype(np.float32))
            et_masks_columns = ['et_mask_' + str(x) for x in range(self.max_et_length-1,-1,-1)]
            et_masks = torch.from_numpy(self.et_mask.loc[idx, et_masks_columns].values.astype(bool))
            if self.wind_speed_path != None:
                wind_speed_columns = ['ws_Pre_' + str(x) for x in range(self.max_et_length-1,-1,-1)]
                wind_speed = torch.from_numpy(self.wind_speed.loc[idx, wind_speed_columns].values.astype(np.float32))
                T_columns = ['T_Pre_' + str(x) for x in range(self.max_et_length-1,-1,-1)]
                temperature= torch.from_numpy(self.temperature.loc[idx, T_columns].values.astype(np.float32))
                P_columns = ['P_Pre_' + str(x) for x in range(self.max_et_length-1,-1,-1)]
                pressure = torch.from_numpy(self.pressure.loc[idx, P_columns].values.astype(np.float32))
                sun_columns = ['sun_Pre_' + str(x) for x in range(self.max_et_length-1,-1,-1)]
                sun_radiance = torch.from_numpy(self.sun_radiance.loc[idx, sun_columns].values.astype(np.float32))
                Rh_columns = ['Rh_Pre_' + str(x) for x in range(self.max_et_length-1,-1,-1)]
                RH = torch.from_numpy(self.relative_humidity.loc[idx, Rh_columns].values.astype(np.float32))
                lai_climate = torch.stack((lai_data, wind_speed, temperature, pressure, sun_radiance, RH),dim=1)
            else:
                lai_climate = lai_data.unsqueeze(0)
            if len(rainfall_event.shape) == 1:
                lai_data = lai_data.unsqueeze(0)
        else:
            et_masks = 0

        rainfall_event = torch.nn.functional.pad(rainfall_event, (0, self.maximum_length - rainfall_event.shape[0]), mode='constant',value=self.padding_value)
        mask = self.create_padding_mask(rainfall_event, self.padding_value)

        if len(rainfall_event.shape) == 1:
            rainfall_event = rainfall_event.unsqueeze(0)


        if self.static_vegetation:
            geo_env_column = ['water_dist', 'fault_dist', 'elev', 'slope','pro_curva', 'plan_curva', 'aspect', 'Coarse_G', 'et_mean']
        else:
            geo_env_column = ['water_dist', 'fault_dist', 'elev', 'slope', 'pro_curva', 'plan_curva', 'aspect',
                              'Coarse_G']
        geo_env = torch.from_numpy(self.geo_env.loc[idx, geo_env_column].values.astype(np.float32))
        # geo_env = geo_env.unsqueeze(0)
        label = self.dataset.loc[idx, 'label']

        return (rainfall_event, lai_climate, mask, et_masks, int(label), geo_env, incid_id)

