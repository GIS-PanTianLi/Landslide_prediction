import torch
from torch import nn
from torch.nn.utils import weight_norm
from Models.ltae import LTAE


class dnn_model(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_class):
        super(dnn_model, self).__init__()
        self.hidden=nn.Sequential(
            nn.Linear(in_channels,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.classfication=nn.Sequential(
            nn.Linear(hidden_dim,num_class),
            nn.Sigmoid()
        )
    def forward(self,x):
        # x = x.squeeze(dim=1)
        x=self.hidden(x)
        x=self.classfication(x)
        return x



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size_list=[2,2], dropout=0.2, seq_len=None, output_dim=None):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            kernel_size = kernel_size_list[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1,
                                     padding=(kernel_size-1), dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.lstm = nn.LSTM(input_size=num_channels[-1], hidden_size=num_channels[-1], num_layers=2, batch_first=True)
        self.fc = nn.Linear(seq_len, output_dim)  # 修正此处维度
        self.gap = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()
    def forward(self, x):
        out = self.network(x)
        return out


class LAI_LSTM(nn.Module):
    def __init__(self, input_dim, mechan_dim, hidden_dim, num_layers):
        super(LAI_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_hydro = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mechan = nn.Sequential(
            nn.Linear(mechan_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_fuse = nn.Linear(hidden_dim*2, hidden_dim)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.gap = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()


    def forward(self, x, et_masks):
        out, _ = self.lstm(x)
        out_mechan = self.fc_mechan(x[:,:,0:1])
        out_hydro = self.fc_hydro(out)
        et_masks = et_masks.unsqueeze(-1)
        out_hydro = out_hydro * et_masks
        out = self.norm(self.fc_fuse(torch.concat([out_mechan, out_hydro], dim=2)).permute(0,2,1))
        out = self.flatten(self.gap(out))
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_length, hidden_size)
        score = self.Va(torch.tanh(self.Wa(lstm_output) + self.Ua(lstm_output))).squeeze(2)
        attention_weights = torch.softmax(score, dim=1)  # 对每个时间步进行softmax
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)  # 加权和

        return context, attention_weights


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=2, keepdim=True)
        max_out, _ = torch.max(x, dim=2, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=2)
        out = self.conv(out.permute(0,2,1))
        return x * self.sigmoid(out).permute(0,2,1), out


class TCN_Landslide(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_geo_env = config['input_geoenv']
        self.input_ante_lai = config['input_lai']
        self.embedd_dim = config['embed_dim']
        self.get_attention = config['return_attention']

        embed_dim = config['embed_dim']
        lai_embed_dim = config['lai_embed_dim']
        self.sp_att = SpatialAttention(kernel_size=9)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=2, batch_first=True)
        self.rainfall_network = TemporalConvNet(num_inputs=1, num_channels=[embed_dim, embed_dim, embed_dim, embed_dim], kernel_size_list=[4, 3, 7, 7],
                                                seq_len=config['input_size_event'], output_dim=config['output_dim'])
        self.temporal_attention = LTAE(in_channels=embed_dim, n_head=config['n_head'], d_k=embed_dim,
                                           d_model=embed_dim, n_neurons=[embed_dim, embed_dim], dropout=config['dropout'],
                                           T=config['input_size_event'], len_max_seq=config['input_size_event'], positions=None, return_att=self.get_attention
                                           )
        if self.input_ante_lai:
            self.lai_network = LAI_LSTM(input_dim=1, hidden_dim=lai_embed_dim, num_layers=3)
        if self.input_geo_env:
            self.geo_envfnn = nn.Sequential(
                nn.Linear(1, config['geo_env_hidden_dim']),
                nn.BatchNorm1d(config['geo_env_dim']),
                nn.ReLU(),
                # nn.Dropout(0.1),
                nn.Linear(config['geo_env_hidden_dim'], config['geo_env_hidden_dim']),
                nn.BatchNorm1d(config['geo_env_dim']),
                nn.ReLU()
            )
            if self.input_ante_lai:
                self.output = dnn_model(in_channels=config['input_size_event']+config['max_et_length']+config['geo_env_dim'],
                                        hidden_dim=config['input_size_event']+config['max_et_length']+config['geo_env_dim'], num_class=config['num_class'])
            else:
                self.output = dnn_model(in_channels=config['input_size_event']+config['geo_env_dim'], hidden_dim=config['input_size_event']+config['geo_env_dim'], num_class=config['num_class'])
        else:
            if self.input_ante_lai:
                self.output = dnn_model(in_channels=config['input_size_event']+config['max_et_length'], hidden_dim=config['input_size_event']+config['max_et_length'], num_class=config['num_class'])
            else:
                self.output = dnn_model(in_channels=config['input_size_event'], hidden_dim=config['input_size_event'], num_class=config['num_class'])

        self.gap = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()


    def forward(self, x, geo_env=None, ante_lai=None, et_masks=None, rainfall_mask=None):
        out = self.rainfall_network(x)
        rainfall_mask = rainfall_mask.unsqueeze(-1)
        out = out.permute(0, 2, 1)
        if self.input_ante_lai:
            lai_out = self.lai_network(ante_lai, et_masks)
            out = torch.concat([out, lai_out], dim=1)

        if self.input_geo_env:
            geoenv_out = self.geo_envfnn(geo_env.permute(0,2,1))
            out = torch.concat([out, geoenv_out], dim=1)
        out, sp_attn = self.sp_att(out)
        out, _ = self.lstm(out)
        out = self.output(self.flatten(self.gap(out).permute(0,2,1)))
        return out, sp_attn



class TCN_Temporal_Attention_Landslide(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_geo_env = config['input_geoenv']
        self.input_ante_lai = config['input_lai']
        self.embedd_dim = config['embed_dim']
        self.get_attention = config['return_attention']

        embed_dim = config['embed_dim']
        lai_embed_dim = config['lai_embed_dim']
        geo_embed_dim = config['geo_env_hidden_dim']
        self.sp_att = SpatialAttention(kernel_size=9)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=2, batch_first=True)

        self.rainfall_network = TemporalConvNet(num_inputs=1, num_channels=[embed_dim, embed_dim, embed_dim, embed_dim], kernel_size_list=[4, 3, 7, 7],
                                                seq_len=config['input_size_event'], output_dim=config['output_dim'])
        self.temporal_attention = LTAE(in_channels=embed_dim, n_head=config['n_head'], d_k=embed_dim,
                                           d_model=embed_dim, n_neurons=[embed_dim, embed_dim], dropout=config['dropout'],
                                           T=config['input_size_event'], len_max_seq=config['input_size_event'], positions=None, return_att=self.get_attention
                                           )

        if self.input_ante_lai:
            self.lai_network = LAI_LSTM(input_dim=6, mechan_dim =1,hidden_dim=lai_embed_dim, num_layers=3)
        if self.input_geo_env:
            self.geo_envfnn = nn.Sequential(
                nn.Linear(config['geo_env_dim'], config['geo_env_hidden_dim']),
                nn.BatchNorm1d(config['geo_env_hidden_dim']),
                nn.ReLU(),
                # nn.Dropout(0.1),
                nn.Linear(config['geo_env_hidden_dim'], config['geo_env_hidden_dim']),
                nn.BatchNorm1d(config['geo_env_hidden_dim']),
                nn.ReLU()
            )
            if self.input_ante_lai:
                self.output = dnn_model(in_channels=embed_dim+lai_embed_dim+geo_embed_dim,
                                        hidden_dim=embed_dim+lai_embed_dim+geo_embed_dim, num_class=config['num_class'])
            else:
                self.output = dnn_model(in_channels=embed_dim+geo_embed_dim, hidden_dim=embed_dim+geo_embed_dim, num_class=config['num_class'])
        else:
            if self.input_ante_lai:
                self.output = dnn_model(in_channels=embed_dim+lai_embed_dim, hidden_dim=embed_dim+lai_embed_dim, num_class=config['num_class'])
            else:
                self.output = dnn_model(in_channels=embed_dim, hidden_dim=embed_dim, num_class=config['num_class'])

        self.gap = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()


    def forward(self, x, geo_env=None, ante_lai=None, et_masks=None, rainfall_mask=None):
        out = self.rainfall_network(x)
        rainfall_mask = rainfall_mask.unsqueeze(-1)
        out = out.permute(0, 2, 1)
        out, att = self.temporal_attention(out)
        if self.input_ante_lai:
            lai_out = self.lai_network(ante_lai, et_masks)
            out = torch.concat([out, lai_out], dim=1)

        if self.input_geo_env:
            geoenv_out = self.geo_envfnn(geo_env)
            out = torch.concat([out, geoenv_out], dim=1)
        f_out = self.output(out)
        if self.input_ante_lai:
            return f_out, [att, out]
        else:

            return f_out, [att, out]