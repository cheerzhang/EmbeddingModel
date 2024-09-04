import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math, os, shutil, pkg_resources
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from datetime import datetime, timedelta
import statsmodels.api as sm


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
    
@staticmethod
def get_version(package_name='emb_model'):
    """
    Get current version
    :param package_name: 
    :return: version
    """
    version = pkg_resources.get_distribution(package_name).version
    return version


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
#            Train Models                      #
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


class CharTransformerModelV2(nn.Module):
    def __init__(self, embN, dimN_dict, nhead, num_layers, max_lens, str_features, num_features, layer_sizes):
        super(CharTransformerModelV2, self).__init__()
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(num_embeddings=embN, embedding_dim=dimN_dict[feature])
            for feature in str_features
        })
        self.pos_encoders = nn.ModuleDict({
            feature: PositionalEncoding(dimN_dict[feature], max_lens[feature])
            for feature in str_features
        })
        self.transformer_encoders = nn.ModuleDict({
            feature: nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=dimN_dict[feature], nhead=nhead),
                num_layers=num_layers
            )
            for feature in str_features
        })
        input_size = sum(dimN_dict.values()) + len(num_features)
        layers = []
        for layer_size in layer_sizes:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            input_size = layer_size
        layers.append(nn.Linear(input_size, 4))  # 输出层，4个类别
        self.classifier = nn.Sequential(*layers)

    def forward(self, str_features, num_features):
        str_feature_outputs = []
        for feature_name, feature_tensor in str_features.items():
            embedded = self.embeddings[feature_name](feature_tensor).permute(1, 0, 2)
            pos_encoded = self.pos_encoders[feature_name](embedded)
            transformer_output = self.transformer_encoders[feature_name](pos_encoded)
            feature_output = transformer_output.mean(dim=0)
            str_feature_outputs.append(feature_output)
        combined_features = torch.cat(str_feature_outputs + list(num_features.values()), dim=1)
        normalized_features = F.normalize(combined_features, p=2, dim=1)
        output = self.classifier(normalized_features)   
        return output
    
class CharTransformerModel(nn.Module):
    def __init__(self, embN, dimN, nhead, num_layers, max_lens, str_features, num_features, layer_sizes):
        super(CharTransformerModel, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=embN, embedding_dim=dimN)
        # 动态创建位置编码器
        self.pos_encoders = nn.ModuleDict({
            feature: PositionalEncoding(dimN, max_lens[feature]) for feature in str_features
        })
        encoder_layers = nn.TransformerEncoderLayer(d_model=dimN, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        layers = []
        input_size = len(str_features) * dimN + len(num_features)
        for layer_size in layer_sizes:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_size = layer_size
        layers.append(nn.Linear(input_size, 4))  # 输出层，4个类别
        self.classifier = nn.Sequential(*layers)

        
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
            'lr': 0.01,
            'char_to_idx': {},
            'str_features': [],
            'num_features': [],
            'layer_sizes' : [2048, 512, 256, 128, 64, 32]
        }
        self.model = None
        self.best_val_loss = float('inf')
    def set_train_parameter(self, batch_size=32, dimN=128, patience=10, lr=0.01,
                            layer_sizes =[2048, 512, 256, 128, 64, 32],
                            best_model_path="best_model.pth"):
        self.train_prameter['batch_size'] = batch_size
        self.train_prameter['dimN'] = dimN
        self.train_prameter['patience'] = patience
        self.train_prameter['lr'] = lr
        self.train_prameter['layer_sizes'] = layer_sizes
        self.best_model_path = best_model_path
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
        self.train_prameter['char_to_idx'] = char_to_idx
        self.train_prameter['str_features'] = str_features
        self.train_prameter['num_features'] = num_features
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
                                          num_features=num_features,
                                          layer_sizes=self.train_prameter['layer_sizes']).to(device)
        class_weights = torch.tensor(self.class_weights, dtype=torch.float).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=self.train_prameter['lr'], weight_decay=1e-4) # Adjust learning rate and weight decay as needed
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.7, verbose=True)
        early_stopping = EarlyStopping(patience=self.train_prameter['patience'], verbose=True)
        num_epochs = 1000 # Define the number of epochs
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
        # eval model
        self.model_eval(self.val_dataloader, char_to_idx, max_len, str_features, num_features)
        return train_losses_list, val_losses_list
    def model_eval(self, val_dataloader, char_to_idx, max_len, str_features, num_features):
        self.model = CharTransformerModel(embN=len(char_to_idx),
                                          dimN=self.train_prameter['dimN'], 
                                          nhead=8, 
                                          num_layers=3, 
                                          max_lens=max_len, 
                                          str_features=str_features, 
                                          num_features=num_features,
                                          layer_sizes=self.train_prameter['layer_sizes'])
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        self.model.to('cpu')
        val_losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_dataloader:
                str_features_batch = {name: batch[i].to('cpu') for i, name in enumerate(str_features)}
                num_features_batch = {name: batch[i + len(str_features)].to('cpu') for i, name in enumerate(num_features)}
                targets = batch[-1].to('cpu').squeeze()
                output = self.model(str_features_batch, num_features_batch)
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        f1 = f1_score(all_targets, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_targets, all_preds)
        # print(f'Validation Loss: {val_loss:.4f}')
        self.metrix = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }
        return self.metrix
    def eval_binary(self, y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().detach().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().detach().numpy()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        results = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "accuracy": accuracy
        }
        return results
    def get_env_file(self):
        # 获取资源文件的路径
        resource_path = pkg_resources.resource_filename(__name__, f'resources/embedding_env.yaml')
        if not os.path.exists(resource_path):
            raise FileNotFoundError(f"Resource file embedding_env.yaml not found in package.")
        shutil.copy(resource_path, os.getcwd())
        print(f"File embedding_env.yaml has been copied to {os.getcwd()}")


class trainModelV2:
    def __init__(self, train_dataset, valid_dataset):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_dataloader = None
        self.val_dataloader = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.train_prameter = {
            'batch_size': 32,
            'nhead': 4,
            'num_layers': 2,
            'patience': 10,
            'lr': 0.01,
            'char_to_idx': {},
            'str_features': [],
            'num_features': [],
            'layer_sizes' : [2048, 512, 256, 128, 64, 32]
        }
        self.model = None
        self.best_val_loss = float('inf')
    def set_train_parameter(self, batch_size=32, dimN_dict={}, patience=10, lr=0.01,
                            nhead=4, num_layers=2,
                            layer_sizes =[2048, 512, 256, 128, 64, 32],
                            best_model_path="best_model.pth"):
        self.train_prameter['batch_size'] = batch_size
        self.train_prameter['dimN_dict'] = dimN_dict
        self.train_prameter['patience'] = patience
        self.train_prameter['lr'] = lr
        self.train_prameter['nhead'] = nhead
        self.train_prameter['num_layers'] = num_layers
        self.train_prameter['layer_sizes'] = layer_sizes
        self.best_model_path = best_model_path
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
        self.train_prameter['char_to_idx'] = char_to_idx
        self.train_prameter['str_features'] = str_features
        self.train_prameter['num_features'] = num_features
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.model = CharTransformerModelV2(embN=len(char_to_idx),
                                          dimN_dict=self.train_prameter['dimN_dict'], 
                                          nhead=self.train_prameter['nhead'],
                                          num_layers=self.train_prameter['num_layers'],
                                          max_lens=max_len, 
                                          str_features=str_features, 
                                          num_features=num_features,
                                          layer_sizes=self.train_prameter['layer_sizes']).to(device)
        class_weights = torch.tensor(self.class_weights, dtype=torch.float).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=self.train_prameter['lr'], weight_decay=1e-4) # Adjust learning rate and weight decay as needed
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.7, verbose=True)
        early_stopping = EarlyStopping(patience=self.train_prameter['patience'], verbose=True)
        num_epochs = 1000 # Define the number of epochs
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
        # eval model
        self.model_eval(self.val_dataloader, char_to_idx, max_len, str_features, num_features)
        return train_losses_list, val_losses_list
    def model_eval(self, val_dataloader, char_to_idx, max_len, str_features, num_features):
        self.model = CharTransformerModelV2(embN=len(char_to_idx),
                                          dimN_dict=self.train_prameter['dimN_dict'], 
                                          nhead=self.train_prameter['nhead'],
                                          num_layers=self.train_prameter['num_layers'],
                                          max_lens=max_len, 
                                          str_features=str_features, 
                                          num_features=num_features,
                                          layer_sizes=self.train_prameter['layer_sizes'])
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        self.model.to('cpu')
        val_losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_dataloader:
                str_features_batch = {name: batch[i].to('cpu') for i, name in enumerate(str_features)}
                num_features_batch = {name: batch[i + len(str_features)].to('cpu') for i, name in enumerate(num_features)}
                targets = batch[-1].to('cpu').squeeze()
                output = self.model(str_features_batch, num_features_batch)
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        f1 = f1_score(all_targets, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_targets, all_preds)
        # print(f'Validation Loss: {val_loss:.4f}')
        self.metrix = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }
        return self.metrix
    def eval_binary(self, y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().detach().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().detach().numpy()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        results = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "accuracy": accuracy
        }
        return results
    def get_env_file(self):
        # 获取资源文件的路径
        resource_path = pkg_resources.resource_filename(__name__, f'resources/embedding_env.yaml')
        if not os.path.exists(resource_path):
            raise FileNotFoundError(f"Resource file embedding_env.yaml not found in package.")
        shutil.copy(resource_path, os.getcwd())
        print(f"File embedding_env.yaml has been copied to {os.getcwd()}")


class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

################################################
#            FE pipeline                       #
################################################
import json
from sklearn.base import BaseEstimator
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
            # check if it is []
            if isinstance(value, list):
                return None
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

class ProcessFilters(BaseEstimator):
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
                X_ = X_[X_[c_name].isin(self.c_values[i])]
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
            if self.period == 'W':
                X_[f"{self.date_column}_{self.period}"] = X_[self.date_column].dt.isocalendar().week
            if self.period == 'w':
                X_[f"{self.date_column}_{self.period}"] = X_[self.date_column].dt.weekday
        return X_


class CheckData(BaseEstimator):
    def __init__(self, check_columns=None, maxlen_columns=None, maxnum_columns=None):
        self.name = 'check_data'
        self.na_inf_result = None
        self.maxlen_columns = maxlen_columns
        self.max_len_result = None
        self.check_columns = check_columns
        self.maxnum_columns = maxnum_columns
        self.max_number_result = None
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
    def max_number_report(self, df, columns):
        X_ = df.copy()
        stats = {}
        for column in columns:
            if column in X_.columns.values:
                value_ = X_[column]
                max_number = X_[column].max()
                q75 = value_.quantile(0.75)
                q90 = value_.quantile(0.90)
                q95 = value_.quantile(0.95)
                q99 = value_.quantile(0.99)
                stats[column] = {'max': max_number, '99q': q99, '95q': q95, '90q': q90, '75q': q75}
            else:
                raise ValueError(f"Missing string feature: {column}")
        return stats
    def fit(self, X, y=None):
        if self.check_columns is not None:
            na_inf_result = self.check_nan_inf(X, self.check_columns)
            self.na_inf_result = na_inf_result
        if self.maxlen_columns is not None:
            max_len = self.max_len_report(X, self.maxlen_columns)
            self.max_len_result = max_len
        if self.maxnum_columns is not None:
            max_v = self.max_number_report(X, self.maxnum_columns)
            self.max_number_result = max_v
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        return X_

   
class ProcessNorm(BaseEstimator):
    def __init__(self, c_name, p_value):
        self.name = 'process_normalization'
        self.c_name = c_name
        self.p_value = p_value
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        if self.c_name not in X_.columns.values:
            raise ValueError(f"Missing string feature: {self.c_name}")  
        else:
            X_[self.c_name] = X_[self.c_name] / self.p_value
        return X_


class MergeDf(BaseEstimator):
    def __init__(self, df, x_on='id', df_on='id', dropDuplicate=False, drop_on='id'):
        self.name = 'process_normalization'
        self.df = df
        self.x_on = x_on
        self.df_on = df_on
        self.dropDuplicate = dropDuplicate
        self.drop_on = drop_on
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        # convert id into same datatype
        if self.x_on not in X_.columns.values:
            raise ValueError(f"Missing feature: {self.x_on} in original dataframe")  
        if self.df_on not in self.df.columns.values:
            raise ValueError(f"Missing feature: {self.df_on} in added in dataframe")  
        X_[self.x_on] = X_[self.x_on].astype(str)
        self.df[self.df_on] = self.df[self.df_on].astype(str)
        if self.dropDuplicate:
            df_merged = df_merged.drop_duplicates(subset=self.drop_on)
        df_merged = X_.merge(self.df, left_on=self.x_on, right_on=self.df_on, how='left')
        return df_merged


class ProcessFeatureInFit(BaseEstimator):
    def __init__(self, feature_name=None):
        self.name = 'process_feature_in_fit'
        self.feature_name = feature_name
        self.feature_value = None
    def fit(self, X, y=None):
        if self.feature_name not in X.columns.values:
            raise ValueError(f"Missing string feature: {self.feature_name}")  
        else:
            self.feature_value = X[self.feature_name].values
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        return X_


class ProcessConCatDF(BaseEstimator):
    def __init__(self, dfs = [], removeDuplicate=False, drop_on=None):
        self.name = 'process_concat_df'
        self.removeDuplicate = removeDuplicate
        self.drop_on = drop_on
        self.dfs = dfs
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        df_new = pd.concat(self.dfs) 
        if self.removeDuplicate:
            df_new = df_new.drop_duplicates(subset=self.drop_on)
        return df_new


def get_5number(value):
    max_number = value.max()
    q75 = value.quantile(0.75)
    q90 = value.quantile(0.90)
    q95 = value.quantile(0.95)
    q99 = value.quantile(0.99)
    return {'max': max_number, '99q': q99, '95q': q95, '90q': q90, '75q': q75}


class MonthlyAnalysis(BaseEstimator):
    def __init__(self, date_column, calulate_column, group_column=[], type='sum'):
        self.name = 'monthly_analysis'
        self.date_column = date_column
        self.group_column = group_column
        self.calulate_column = calulate_column
        self.type = type
        self.monthly = None
    def fit(self, X, y=None):
        X['year_month'] = X[self.date_column].dt.to_period('M')
        if self.type == 'sum':
            self.monthly = X.groupby(self.group_column + ['year_month'])[self.calulate_column].sum().reset_index()
        if self.type == 'mean':
            self.monthly = X.groupby(self.group_column + ['year_month'])[self.calulate_column].mean().reset_index()
        if self.type == 'TF':
            X['is_none'] = X[self.calulate_column].isna()
            self.monthly = X.groupby(self.group_column + ['year_month'])['is_none'].any().reset_index()
        return self
    def transform(self, X, y=None):
        return X


class GroupAnalysis(BaseEstimator):
    def __init__(self, calulate_column, group_column=[], type='TF'):
        self.name = 'monthly_analysis'
        self.group_column = group_column
        self.calulate_column = calulate_column
        self.type = type
        self.df = None
    def fit(self, X, y=None):
        if self.type == 'TF':
            X['is_none'] = X[self.calulate_column].isna()
            self.df = X.groupby(self.group_column)['is_none'].any().reset_index()
        return self
    def transform(self, X, y=None):
        return X



def get_week_range(date, days=7):
    """ input a date, return the start week date and end week date """
    start_of_week = date - timedelta(days=date.weekday())
    end_of_week = start_of_week + timedelta(days=days)
    start_of_week = pd.Timestamp(start_of_week).replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_week = pd.Timestamp(end_of_week).replace(hour=0, minute=0, second=0, microsecond=0)
    return start_of_week, end_of_week


def get_week_starts(df, date_column):
    """ input a df, return the week start days array of the df """
    df[date_column] = pd.to_datetime(df[date_column])
    df['week_start'] = (df[date_column] - pd.to_timedelta(df[date_column].dt.weekday, unit='d')).dt.date
    week_starts = df['week_start'].drop_duplicates().sort_values().reset_index(drop=True)
    return week_starts


class ABTestRatio(BaseEstimator):
    def __init__(self, type='Count', counts=[100, 100, 100, 100], dfs=[None, None], column=''):
        self.name = 'AB_Test'
        self.type = type  # DF or Count
        self.counts = counts
        self.dfs = dfs
        self.column = column
        self.alpha = 0.05
        self.status_increase = True
        self.status_drop = True
        self.status_same = True
        self.result = "DK"

    def config(self):
        if self.type == 'Count':
            self.A_Pass_Count = self.counts[0]
            self.A_Fail_Count = self.counts[1]
            self.B_Pass_Count = self.counts[2]
            self.B_Fail_Count = self.counts[3]
        elif self.type == 'DF':
            A_df = self.dfs[0]
            self.A_Pass_Count = A_df[A_df[self.column] == True].shape[0]
            self.A_Fail_Count = A_df[A_df[self.column] == False].shape[0]
            B_df = self.dfs[1]
            self.B_Pass_Count = B_df[B_df[self.column] == True].shape[0]
            self.B_Fail_Count = B_df[B_df[self.column] == False].shape[0]
        self.A_total = self.A_Pass_Count + self.A_Fail_Count
        self.B_total = self.B_Pass_Count + self.B_Fail_Count

    def p0_B_sg_A(self):
        count = np.array([self.A_Pass_Count, self.B_Pass_Count])
        nobs = np.array([self.A_total, self.B_total])
        z_stat, p_value = sm.stats.proportions_ztest(count, nobs, alternative='smaller')
        if p_value < self.alpha:
            self.status_drop = False

    def p0_B_lg_A(self):
        count = np.array([self.A_Pass_Count, self.B_Pass_Count])
        nobs = np.array([self.A_total, self.B_total])
        z_stat, p_value = sm.stats.proportions_ztest(count, nobs, alternative='larger')
        if p_value < self.alpha:
            self.status_increase = False

    def p0_B_same_A(self):
        count = np.array([self.A_Pass_Count, self.B_Pass_Count])
        nobs = np.array([self.A_total, self.B_total])
        z_stat, p_value = sm.stats.proportions_ztest(count, nobs, alternative='two-sided')
        if p_value < self.alpha:
            self.status_same = False

    def fit(self, X, y=None):
        self.config()
        if self.A_total == 0 or self.B_total == 0:
            self.result = "No data"
        else:
            self.p0_B_sg_A()
            self.p0_B_lg_A()
            self.p0_B_same_A()

            # 设置 result 的逻辑
            if not self.status_same:
                if self.status_drop and not self.status_increase:
                    self.result = "Decrease"
                elif not self.status_drop and self.status_increase:
                    self.result = "Increase"
                else:
                    self.result = "Different"
            else:
                self.result = "quite Same"

        return self

    def transform(self, X, y=None):
        return X


class FilterRange(BaseEstimator):
    def __init__(self, column='', range_min=None, range_max=None, type='left'):
        self.name = 'filter_by_range'
        self.column = column
        self.range_min = range_min
        self.range_max = range_max
        self.type = type # left >= & < ; right > % <= ; all: >= and <= ; none: > & <
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        if self.column not in X_.columns.values:
            raise ValueError(f"Missing string feature: {self.feature_name}")  
        else:
            if self.type == 'left':
                X_ = X_[(X_[self.column]>=self.range_min) & (X_[self.column]<self.range_max)]
            if self.type == 'right':
                X_ = X_[(X_[self.column]>self.range_min) & (X_[self.column]<=self.range_max)]
            if self.type == 'all':
                X_ = X_[(X_[self.column]>=self.range_min) & (X_[self.column]<=self.range_max)]
            if self.type == 'none':
                X_ = X_[(X_[self.column]>self.range_min) & (X_[self.column]<self.range_max)]
        return X_
    

# same item in array mapping
# for df['same_item'] is an array for item_id, map all the same item id into map
def update_item_mapping(item, df, same_item_map, same_item_name = 'same_item'):
    df_ = df.copy()
    filtered_df = df_[df_[same_item_name].apply(lambda x: item in x.split(','))]
    # get merged items list
    merged_ = []
    for items in filtered_df[same_item_name]:
        merged_.extend(items.split(','))
        merged_ = list(set(merged_))
    # insert or update
    if item not in same_item_map.keys() and all(item not in values for values in same_item_map.values()):
        same_item_map[item] = merged_
    else:
        existing_key = None
        for key, values in same_item_map.items():
            if item in values:
                existing_key = key
        if existing_key:
            current_ = same_item_map[existing_key]
            new_ = merged_
            same_item_map[existing_key] = list(set(current_ + new_)) 
    return same_item_map


def update_map_from_another_map(map1, map2, df, column_map1 = 'same_item', column_map2 = 'id'):
    df_ = df.copy()
    for key, value in map1.items():
        filtered_df = df_[df_[column_map1].apply(lambda x: key in x.split(','))]
        if key not in map2.keys():
            map2[key] = list(set(filtered_df[column_map2].values.tolist()))
        else:
            current_uuids = map2[key]
            new_uuids = filtered_df[column_map2].values.tolist()
            map2[key] = list(set(current_uuids + new_uuids))  
    return map2



#######################################
#                   XGB               #
#######################################
import xgboost as xgb


class trainXGBbinary:
    def __init__(self):
        self.name = 'xgb bianry classification training'
        self.model = None
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': 0.1,
            'max_depth': 6,
            'lambda': 1.0,     # L2
            'alpha': 0.1,      # L1
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1  # weight
        }
    def config_train_parameter(self, lambda_ = 1.0, alpha = 0.1):
        self.params['lambda'] = lambda_
        self.params['alpha'] = alpha
        return self.params
    def train(self, train_df, valid_df, features, label):
        neg, pos = np.bincount(train_df[label])
        scale_pos_weight = neg / pos
        self.params['scale_pos_weight'] = scale_pos_weight 
        dtrain = xgb.DMatrix(train_df[features], label=train_df[label])
        dvalid = xgb.DMatrix(valid_df[features], label=valid_df[label])
        evallist = [(dtrain, 'train'), (dvalid, 'eval')]
        num_round = 1000
        evals_result = {}
        bst = xgb.train(self.params, dtrain, num_round, evallist, evals_result=evals_result, early_stopping_rounds=10)
        self.model = bst
        return evals_result, bst
    def eval_model(self, valid_df, features, label):
        dvalid = xgb.DMatrix(valid_df[features], label=valid_df[label])
        y_pred_proba = self.model.predict(dvalid)
        y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_proba]  # set as 0.5 
        accuracy, precision, recall, f1, auc = self.calculate_metrics(valid_df[label].values, y_pred, y_pred_proba)
        result = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC': auc
        }
        return result
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba)
        return accuracy, precision, recall, f1, auc



class trainXGBregression:
    def __init__(self):
        self.name = 'xgb regression training'
        self.model = None
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.1,
            'max_depth': 6,
            'lambda': 1.0,     # L2
            'alpha': 0.1,      # L1
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    def correlation(self, df, features, label):
        df_ = df[features + [label]].copy()
        correlation_matrix = df_[features + [label]].corr()
        correlation_with_label = correlation_matrix[label].drop(label)
        correlation_df = correlation_with_label.to_frame()
        return correlation_df
    def config_train_parameter(self, lambda_ = 1.0, alpha = 0.1):
        self.params['lambda'] = lambda_
        self.params['alpha'] = alpha
        return self.params
    def train(self, train_df, valid_df, features, label):
        dtrain = xgb.DMatrix(train_df[features], label=train_df[label])
        dvalid = xgb.DMatrix(valid_df[features], label=valid_df[label])
        evallist = [(dtrain, 'train'), (dvalid, 'eval')]
        num_round = 1000
        evals_result = {}
        bst = xgb.train(self.params, dtrain, num_round, evallist, evals_result=evals_result, early_stopping_rounds=10)
        self.model = bst
        return evals_result, bst
    def eval_model(self, valid_df, features, label):
        dvalid = xgb.DMatrix(valid_df[features], label=valid_df[label])
        y_pred = self.model.predict(dvalid)
        mse, mae, mape, rmse = self.calculate_metrics(valid_df[label].values, y_pred)
        result = {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse
        }
        return result
    def calculate_metrics(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mse = np.mean((y_pred - y_true) ** 2)
        mae = np.mean(np.abs(y_pred - y_true))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        rmse = np.sqrt(mse)
        return mse, mae, mape, rmse


########################################
#            GRU  LSTM                 #
########################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class trainGRUregression:
    def __init__(self, n_steps = 5):
        self.name = 'train GRU regression'
        self.n_step = n_steps
        self.batch_size = 32
        self.model = None
        self.best_model_path = 'gru_model.pth'
        self.parameter = {
           'hidden_dim': 64,
           'num_layers' : 2
        }
    def create_sequences(self, df, features, label, gourp_id):
        df_ = df.copy()
        f_ = features + [label]
        sequences = []
        labels = []
        for group_id, group_data in df_.groupby(gourp_id):
            group_data = group_data[f_].values
            for i in range(len(group_data) - self.n_step):
                sequence = group_data[i:i + self.n_step, :-1]  # feature
                label = group_data[i + self.n_step, -1]  # label
                sequences.append(sequence)
                labels.append(label)
        return np.array(sequences), np.array(labels)
    def setconfig(self, hidden_dim=64, num_layers=2):
        self.parameter['hidden_dim'] = hidden_dim
        self.parameter['num_layers'] = num_layers
        return self.parameter
    def train(self, train_df, valid_df, features, label, gourp_id):
        X_train, y_train = self.create_sequences(train_df, features, label, gourp_id)
        X_valid, y_valid = self.create_sequences(valid_df, features, label, gourp_id)
        print(f"Shape of train X: {X_train.shape}, Shape of train y: {y_train.shape}")
        print(f"Shape of valid X: {X_valid.shape}, Shape of valid y: {y_valid.shape}")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
        y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        input_dim = X_train.shape[2]
        hidden_dim = self.parameter['hidden_dim']
        output_dim = 1
        num_layers = self.parameter['num_layers']
        self.model = GRUModel(input_dim, hidden_dim, output_dim, num_layers)
        best_val_loss = float('inf')
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)       
        
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.model.to(device)
        num_epochs = 1000
        patience = 10
        best_valid_loss = np.inf
        epochs_no_improve = 0
        train_loss_array = []
        valid_loss_array = []
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_loss_array.append(train_loss)
            self.model.eval() 
            valid_loss = 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets.unsqueeze(1))
                    valid_loss += loss.item()
            valid_loss /= len(valid_loader)
            valid_loss_array.append(valid_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
            scheduler.step(valid_loss)
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"Best model saved with validation loss: {valid_loss:.6f}")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        self.model.eval()
        with torch.no_grad():
            test_loss = 0
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                test_loss += loss.item()
        test_loss /= len(train_loader)
        print(f'Eval model Loss: {test_loss:.4f}')
        return train_loss_array, valid_loss_array
    def create_sequences_for_prediction(self, df, features, label, group_id):
        df_ = df.copy()
        f_ = features + [label]
        sequences = []
        indices = []
        for group_id, group_data in df_.groupby(group_id):
            group_data = group_data[f_].copy()
            group_data_index = group_data.index
            group_data = group_data.values
            for i in range(len(group_data) - self.n_step):
                sequence = group_data[i:i + self.n_step, :-1]  # 前 n_steps 天的数据作为 feature
                sequences.append(sequence)
                indices.append(group_data_index[i + self.n_step])  # 保存真实的索引
        return np.array(sequences), indices
    def predict(self, test_df, features_, label, group_id):

        new_data = test_df.copy()
        X_new, indices = self.create_sequences_for_prediction(new_data, features_, label, group_id)
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32).to('cpu')
        self.model.to('cpu')
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_new_tensor).numpy()
        return indices, predictions
    def predict_model(self, test_df, features_, label, group_id):
        new_data = test_df.copy()
        predicted_df = new_data.iloc[self.n_step:].copy()
        predicted_df['predicted_label'] = np.nan
        indices, predictions = self.predict(test_df, features_, label, group_id)
        for idx, prediction in zip(indices, predictions):
            predicted_df.loc[idx, 'predicted_label'] = prediction
        return predicted_df


