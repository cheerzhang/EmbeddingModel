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
            stats[column] = {'max': max_len, '90q': q90, '75q': q75}
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

    