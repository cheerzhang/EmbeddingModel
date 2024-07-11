import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math


class CDataset(Dataset):
    def __init__(self, 
        str_features, num_features,
        target, char_to_idx, max_lens):
        """
        str_features: List of lists, where each sublist is a string feature column
        num_features: List of lists, where each sublist is a numeric feature column
        target: List, the target column
        char_to_idx: Dict, a mapping from character to index
        max_lens: Dict, a mapping from feature name to maximum length
        """
        self.str_features = {name: features for name, features in zip(max_lens.keys(), str_features)}
        self.num_features = num_features
        self.target = target
        self.char_to_idx = char_to_idx
        self.max_lens = max_lens
    def __len__(self):
        return len(self.target)
    def __getitem__(self, idx):
        str_features_encoded = []
        for feature_name, str_feature in self.str_features.items():
            encode = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in str_feature[idx]]
            padded_encode = encode[:self.max_lens[feature_name]] + [self.char_to_idx['<PAD>']] * max(0, self.max_lens[feature_name] - len(encode))
            str_features_encoded.append(torch.tensor(padded_encode, dtype=torch.long))
        num_features_encoded = [torch.tensor([num_feature[idx]], dtype=torch.float) for num_feature in self.num_features]

        return (*str_features_encoded, *num_features_encoded, torch.tensor([self.target[idx]], dtype=torch.float))
    
        

def max_len_report(df, columns):
    X_ = df.copy()
    stats = {}
    for column in columns:
        if column in X_.columns.values:
            lengths = X_[column].apply(len)
            max_len = lengths.max()
            q75 = lengths.quantile(0.75)
            q90 = lengths.quantile(0.90)
            q95 = lengths.quantile(0.95)
            q99 = lengths.quantile(0.99)
            stats[column] = {'max': max_len, '99q': q99, '95q': q95, '90q': q90, '75q': q75}
        else:
            raise ValueError(f"Missing string feature: {column}")
    return stats


def create_char_to_idx(texts, special_tokens=['<PAD>', '<UNK>']):
    chars = set(''.join(texts))
    char_to_idx = {char: idx + len(special_tokens) for idx, char in enumerate(chars)}
    for idx, token in enumerate(special_tokens):
        char_to_idx[token] = idx
    return char_to_idx  

################################################
#            Traom Models                      #
################################################
class PositionalEncoding(nn.Module):
    def __init__(self, dimN, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dimN)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dimN, 2).float() * (-math.log(10000.0) / dimN))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class CharTransformerModel(nn.Module):
    def __init__(self, embN, dimN, nhead, num_layers, max_lens, str_features, num_features):
        super(CharTransformerModel, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=embN, embedding_dim=dimN)
        # 动态创建位置编码器
        self.pos_encoders = nn.ModuleDict({
            feature: PositionalEncoding(dimN, max_lens[feature]) for feature in str_features
        })
        encoder_layers = nn.TransformerEncoderLayer(d_model=dimN, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(11*dimN + 11, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4), 
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 4)  # 4个类别
        )
    def forward(self, str_features, num_features):
        str_feature_outputs = []
        # 处理每个字符串特征
        for feature_name, feature_tensor in str_features.items():
            embedded = self.embeddings(feature_tensor).permute(1, 0, 2)
            pos_encoded = self.pos_encoders[feature_name](embedded)
            transformer_output = self.transformer_encoder(pos_encoded)
            feature_output = transformer_output.mean(dim=0)
            str_feature_outputs.append(feature_output)

        # 将所有字符串特征和数值特征连接在一起
        combined_features = torch.cat(str_feature_outputs + list(num_features.values()), dim=1)
        normalized_features = F.normalize(combined_features, p=2, dim=1)
        output = self.classifier(normalized_features)   
        return output
        
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class trainModel:
    def __init__(self, train_dataset, valid_dataset):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_dataloader = None
        self.val_dataloader = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.train_prameter = {
            'batch_size': 32,
            'dimN': 128,
            'patience': 10,
            'lr': 0.01
        }
        self.model = None
    def set_train_parameter(self, batch_size=32, dimN=128, patience=10, lr=0.01):
        self.train_prameter['batch_size'] = batch_size
        self.train_prameter['dimN'] = dimN
        self.train_prameter['patience'] = patience
        self.train_prameter['lr'] = lr
        return self.train_prameter
    def prepare_data(self):
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.train_prameter['batch_size'], shuffle=True)
        self.val_dataloader = DataLoader(self.valid_dataset, batch_size=self.train_prameter['batch_size'], shuffle=False)
    def get_class_weight(self, df, label_columns):
        labels = np.array(df[label_columns].values)
        class_sample_counts = np.bincount(labels)
        total_samples = len(labels)
        class_weights = total_samples / (len(class_sample_counts) * class_sample_counts)
        self.class_weights = class_weights
        return class_weights
    def train(self, char_to_idx, max_len, str_features, num_features):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.model = CharTransformerModel(embN=len(char_to_idx),
                                          dimN=self.train_prameter['dimN'], 
                                          nhead=8, 
                                          num_layers=3, 
                                          max_lens=max_len, 
                                          str_features=str_features, 
                                          num_features=num_features).to(device)
        class_weights = torch.tensor(self.class_weights, dtype=torch.float).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=self.train_prameter['lr'], weight_decay=1e-4) # Adjust learning rate and weight decay as needed
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.7, verbose=True)
        early_stopping = EarlyStopping(patience=self.train_prameter['patience'], verbose=True)
        num_epochs = 10 # Define the number of epochs
        train_losses_list = []
        val_losses_list = []
        for epoch in range(num_epochs):
            self.model.train()
            train_losses = []
            for batch in self.train_dataloader:
                str_features_batch = {name: batch[i].to(device) for i, name in enumerate(str_features)}
                num_features_batch = {name: batch[i + len(str_features)].to(device) for i, name in enumerate(num_features)}
                targets = batch[-1].to(device).squeeze()
                optimizer.zero_grad()
                output = self.model(str_features_batch, num_features_batch)
                loss = self.criterion(output, targets)  # Assuming `targets` is provided
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = sum(train_losses) / len(train_losses)
            train_losses_list.append(train_loss)
            # eval
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in self.val_dataloader:
                    str_features_batch = {name: batch[i].to(device) for i, name in enumerate(str_features)}
                    num_features_batch = {name: batch[i + len(str_features)].to(device) for i, name in enumerate(num_features)}
                    targets = batch[-1].to(device).squeeze()
                    output = self.model(str_features_batch, num_features_batch)
                    loss = self.criterion(output, targets)
                    val_losses.append(loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            val_losses_list.append(val_loss)
            # check if this is the best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"Best model saved with validation loss: {val_loss:.6f}")
            # early stop
            if epoch % 1 == 0:
                print(f'Epoch {epoch}, Loss: {train_loss}, {val_loss}')
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # Update the learning rate
            scheduler.step(val_loss)
        self.train_losses_list = train_losses_list
        self.val_losses_list = val_losses_list
        return train_losses_list, val_losses_list




################################################
#            FE pipeline                       #
################################################
import json
from sklearn.base import BaseEstimator
from datetime import datetime
import numpy as np

class ProcessJson(BaseEstimator):
    def __init__(self, nodes, c_names, f_names):
        self.name = 'process_json'
        self.nodes = nodes
        self.c_names = c_names
        self.f_names = f_names
    def fit(self, X, y=None):
        return self
    def process_json(self, row, key_names):
        try:
            parsed_json = json.loads(row)
            value = parsed_json
            for key in key_names:
                if key in value:
                    value = value[key]
                else:
                    return None
            return value
        except (json.JSONDecodeError, TypeError, KeyError):
            return None
    def get_feature_from_json(self, df, json_column_name, key_names):
        return df[json_column_name].apply(self.process_json, args=(key_names,))
    def transform(self, X, y=None):
        X_ = X.copy()
        for i, node in enumerate(self.nodes):
            if self.c_names[i] in X_.columns.values:
                X_[self.f_names[i]] = self.get_feature_from_json(X_, self.c_names[i], node)
            else:
                raise ValueError(f"Missing string feature: {self.c_names[i]}")
        return X_

class ProcessFilter(BaseEstimator):
    def __init__(self, c_names, c_values):
        self.name = 'process_filter'
        self.c_names = c_names
        self.c_values = c_values
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        for i, c_name in enumerate(self.c_names):
            if c_name in X_.columns.values:
                X_ = X_[X_[c_name]==self.c_values[i]]
            else:
                raise ValueError(f"Missing string feature: {c_name}")
        return X_

class ProcessStr(BaseEstimator):
    def __init__(self, c_names):
        self.name = 'process_str'
        self.c_names = c_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        for i, c_name in enumerate(self.c_names):
            if c_name in X_.columns.values:
                X_[c_name] = X_[c_name].fillna('')
                X_[c_name] = X_[c_name].astype(str).str.lower().str.strip().str.replace("\s+", "", regex=True)
            else:
                raise ValueError(f"Missing string feature: {c_name}")
        return X_


class ProcessNumer(BaseEstimator):
    def __init__(self, c_names):
        self.name = 'process_number'
        self.c_names = c_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        for i, c_name in enumerate(self.c_names):
            if c_name in X_.columns.values:
                X_[c_name] = pd.to_numeric(X_[c_name], errors='coerce')
                X_[c_name] = X_[c_name].fillna(-1)
            else:
                raise ValueError(f"Missing string feature: {c_name}")
        return X_


class ProcessAge(BaseEstimator):
    def __init__(self, c_birthdate):
        self.name = 'process_age'
        self.c_name = c_birthdate
    def fit(self, X, y=None):
        return self
    def calculate_age_(self, row):
        current_date = datetime.now()
        age = current_date.year - row[self.c_name].year
        if row[self.c_name].month > current_date.month:
            age -= 1
        return age
    def transform(self, X, y=None):
        X_ = X.copy()
        if self.c_name in X_.columns.values:
            X_["age"] = X_.apply(self.calculate_age_, axis=1)
            X_["age"] = X_["age"].fillna(-1)
        else:
            raise ValueError(f"Missing string feature: {self.c_name}")
        return X_
    

import pandas as pd
class PrcocessDate(BaseEstimator):
    def __init__(self, c_dates):
        self.name = 'process_date'
        self.c_dates = c_dates
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        for i, c_name in enumerate(self.c_dates):
            if c_name in X_.columns.values:
                X_[c_name] = pd.to_datetime(X_[c_name], errors='coerce')
            else:
                raise ValueError(f"Missing string feature: {c_name}")
        return X_


class ProcessCombineFE(BaseEstimator):
    def __init__(self, c_names, n_name):
        self.name = 'process_combine_fe'
        self.c_names = c_names
        self.n_name = n_name
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        X_[self.n_name] = ''
        for i, c_name in enumerate(self.c_names):
            if c_name not in X_.columns.values:
                raise ValueError(f"Missing string feature: {c_name}")
            else:
                X_[self.n_name] = X_[self.n_name] + ' ' + X_[c_name]
        X_[self.n_name] = X_[self.n_name].apply(lambda x: x[1:] if isinstance(x, str) and len(x) > 0 else x)
        return X_


class ProcessSplitFE(BaseEstimator):
    def __init__(self, c_name, n_name, s_split, n_part, fillna=None):
        self.name = 'process_split_fe'
        self.c_name = c_name
        self.s_split = s_split
        self.n_part = n_part
        self.n_name = n_name
        self.fillna = fillna
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        if self.c_name not in X_.columns.values:
            raise ValueError(f"Missing string feature: {self.c_name}")
        else:
            X_[self.n_name] = X_[self.c_name].apply(lambda x: x.split(self.s_split)[self.n_part] if len(x.split(self.s_split)) > self.n_part else self.fillna)
        return X_


class ProcessDInDate(BaseEstimator):
    def __init__(self, date_column, period):
        self.name = 'get_date_from_date'
        self.date_column = date_column
        self.period = period
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        if self.date_column not in X_.columns.values:
            raise ValueError(f"Missing string feature: {self.date_column}")   
        else:
            if self.period == 'D':
                X_[f"{self.date_column}_{self.period}"] = X_[self.date_column].dt.day
            if self.period == 'M':
                X_[f"{self.date_column}_{self.period}"] = X_[self.date_column].dt.month
            if self.period == 'Y':
                X_[f"{self.date_column}_{self.period}"] = X_[self.date_column].dt.year
            if self.period == 'h':
                X_[f"{self.date_column}_{self.period}"] = X_[self.date_column].dt.hour
            if self.period == 'w':
                X_[f"{self.date_column}_{self.period}"] = X_[self.date_column].dt.weekday
        return X_


class CheckData(BaseEstimator):
    def __init__(self, check_columns=None, max_columns=None):
        self.name = 'check_data'
        self.na_inf_result = None
        self.max_columns = max_columns
        self.max_len_result = None
        self.check_columns = check_columns
    def check_nan_inf(self, df, columns):
        result = {}
        for col in columns:
            nans = df[col].isna().sum()
            infs = 0
            if pd.api.types.is_numeric_dtype(df[col]):
                infs = np.isinf(df[col]).sum()
            if nans > 0 or infs > 0:
                result[col] = {'NaN': nans, 'Inf': infs}
        return result
    def max_len_report(self, df, columns):
        X_ = df.copy()
        stats = {}
        for column in columns:
            if column in X_.columns.values:
                lengths = X_[column].apply(len)
                max_len = lengths.max()
                q75 = lengths.quantile(0.75)
                q90 = lengths.quantile(0.90)
                q95 = lengths.quantile(0.95)
                q99 = lengths.quantile(0.99)
                stats[column] = {'max': max_len, '99q': q99, '95q': q95, '90q': q90, '75q': q75}
            else:
                raise ValueError(f"Missing string feature: {column}")
        return stats
    def fit(self, X, y=None):
        if self.check_columns is not None:
            na_inf_result = self.check_nan_inf(X, self.check_columns)
            self.na_inf_result = na_inf_result
        if self.max_columns is not None:
            max_len = self.max_len_report(X, self.max_columns)
            self.max_len_result = max_len
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        return X_