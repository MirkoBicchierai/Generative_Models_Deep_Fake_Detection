import os
from os.path import isfile
import torch
from torch.utils.data import Dataset
import numpy as np


class FastDataset(Dataset):
    def __init__(self, folder_path, label_target, one_vs_rest):
        self.label_target = label_target
        self.folder_path = folder_path
        self.one_vs_rest = one_vs_rest
        self.file_list = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path))
            if isfile(os.path.join(folder_path, f))
            and (any(label in f for label in self.label_target))
        ]

        self.labels = []
        for f in self.file_list:
            file_name = os.path.basename(f)
            label = next(
                (i for i, target in enumerate(self.label_target) if target in file_name)
            )
            self.labels.append(label)

        if self.one_vs_rest:
            self.labels = [1 if x != 0 else 0 for x in self.labels]
            self.num_classes = 2
        else:
            self.num_classes = len(self.label_target)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        sequence = np.load(self.file_list[idx], allow_pickle=True)
        label = self.labels[idx]

        sequence = torch.Tensor(sequence)
        return sequence.squeeze(), label


class FlattenedMLPDataset(Dataset):
    def __init__(self, folder_path, label_target, one_vs_rest):
        self.label_target = label_target
        self.folder_path = folder_path
        self.one_vs_rest = one_vs_rest
        self.file_list = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path))
            if isfile(os.path.join(folder_path, f))
            and (any(label in f for label in self.label_target))
        ]

        self.data = []
        for f in self.file_list:
            sequence = np.load(f, allow_pickle=True)
            label = next(
                (
                    i
                    for i, target in enumerate(self.label_target)
                    if target in os.path.basename(f)
                )
            )
            if self.one_vs_rest:
                label = 1 if label != 0 else 0

            self.data.extend([(vector, label) for vector in sequence])

        self.num_classes = 2 if self.one_vs_rest else len(self.label_target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vector, label = self.data[idx]
        vector = torch.Tensor(vector).squeeze(0)
        return vector, label
