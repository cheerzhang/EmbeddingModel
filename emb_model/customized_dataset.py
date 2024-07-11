import torch
from torch.utils.data import DataLoader, Dataset


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
        self.str_features = str_features
        self.num_features = num_features
        self.target = target
        self.char_to_idx = char_to_idx
        self.max_lens = max_lens
    def __len__(self):
        return len(self.target)
    def __getitem__(self, idx):
        str_features_encoded = []
        for i, str_feature in enumerate(self.str_features):
            encode = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in str_feature[idx]]
            padded_encode = encode[:self.max_lens[i]] + [self.char_to_idx['<PAD>']] * max(0, self.max_lens[i] - len(encode))
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