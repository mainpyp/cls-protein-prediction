import h5py
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset, DataLoader


class TMH(TensorDataset):
    def __init__(self, embeddings_path, protein_hashes_path, train_ids):

        self.label_mappings = {
            'G_SP': 0,
            'G': 1,
            'SP_TM': 2,
            'TM': 3
        }
        self.reverse_label_mappings = {val: key for key, val in self.label_mappings.items()}

        embeddings_prot = h5py.File(embeddings_path, "r")

        with open(protein_hashes_path, 'rb') as handle:
            proteins_and_hashes = pickle.load(handle)

        label_prot = pd.read_csv(train_ids)

        X_train, X_test = train_test_split(label_prot, 
        test_size=0.3, 
        train_size=0.7, 
        random_state=42, 
        stratify=label_prot["label"])

        embeddings = []
        labels = []
        protein_ids = []
        for index, row in X_train.iterrows():
            hash_code = proteins_and_hashes[row["label"]][row["prot_id"]][2]
            if hash_code == '':
                continue
            mean_embedding = np.mean(embeddings_prot.get(hash_code), axis=0)
            embeddings.append(mean_embedding)
            labels.append(row["label"])
            protein_ids.append(row["prot_id"])

        embeddings = torch.Tensor(embeddings)
        labels = torch.Tensor([self.label_mappings[label] for label in  labels])
        labels = one_hot(labels.to(torch.int64), 4)

        super().__init__(embeddings, labels)