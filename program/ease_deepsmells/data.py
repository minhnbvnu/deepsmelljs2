# pytorch lib
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ds lib
import numpy as np

# scikit-learn lib
from sklearn.utils import compute_class_weight


class Dataset(Dataset):
    def __init__(self, inputs, labels, is_test=False):
        self.is_test = is_test
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        # sample_input = self.inputs[idx]
        sample_input = torch.tensor(self.inputs[idx], dtype=torch.int)
        if self.is_test:
            return sample_input
        else:
            sample_label = torch.tensor([self.labels[idx]], dtype=torch.int)
            # sample_label = self.labels[idx]
            return (sample_input, sample_label)

    def __len__(self):
        return self.inputs.shape[0]
