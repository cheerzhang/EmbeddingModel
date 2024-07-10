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
    
        
