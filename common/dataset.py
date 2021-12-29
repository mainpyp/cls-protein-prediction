import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader


class TMHLoader(Dataset):
    def __init__(self, df, labels):
        super().__init__()
        self.df = df
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.df[item], self.labels[item]

class TMH(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.label_mappings = {
            'G_SP': 0,
            'G': 1,
            'SP_TM': 2,
            'TM': 3
        }
        self.reverse_label_mappings = {val: key for key, val in self.label_mappings.items()}

    def prepare_data(self):
        root = Path(self.cfg.data_root)
        embeddings_prot = h5py.File(root / "embeddings.h5", "r")

        with open(root / "seq_anno_hash.pickle", 'rb') as handle:
            proteins_and_hashes = pickle.load(handle)

        all_labels = pd.read_csv(root / "data_splits/train_prot_id_labels.csv")
        X_test = pd.read_csv(root / "data_splits/test_prot_id_labels.csv")

        X_train, X_val = train_test_split(all_labels,
                                           test_size=0.3,
                                           train_size=0.7,
                                           random_state=42,
                                           stratify=all_labels["label"])

        train_embed, train_labels = self._parse(X_train, proteins_and_hashes, embeddings_prot)
        val_embed, val_labels = self._parse(X_val, proteins_and_hashes, embeddings_prot)
        test_embed, test_labels = self._parse(X_test, proteins_and_hashes, embeddings_prot)

        self.embeddings = {
            "train": train_embed,
            "val": val_embed,
            "test": test_embed
        }
        self.labels = {
            "train": train_labels,
            "val": val_labels,
            "test": test_labels
        }

    def _parse(self, split, proteins_and_hashes, embeddings_prot):
        # TODO: optimize this whole method
        embeddings = []
        labels = []
        protein_ids = []
        for index, row in split.iterrows():
            hash_code = proteins_and_hashes[row["label"]][row["prot_id"]][2]
            if hash_code == '':
                continue
            mean_embedding = np.mean(embeddings_prot.get(hash_code), axis=0)
            embeddings.append(mean_embedding)
            labels.append(row["label"])
            protein_ids.append(row["prot_id"])

        labels = torch.Tensor([self.label_mappings[label] for label in labels])
        labels = one_hot(labels.to(torch.int64), 4)
        return torch.Tensor(embeddings), labels

    def _dataloader(self, mode):
        dataloader = TMHLoader(df=self.embeddings[mode], labels=self.labels[mode])
        return DataLoader(dataset=dataloader,
                          shuffle=mode == "train",
                          num_workers=0, #TODO: load from cfg
                          batch_size=2)

    def train_dataloader(self):
        return self._dataloader(mode="train")

    def val_dataloader(self):
        return self._dataloader(mode="val")

    def test_dataloader(self):
        return self._dataloader(mode="test")