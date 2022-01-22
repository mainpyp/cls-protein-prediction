import hashlib
from io import StringIO
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from dnachisel.biotools import reverse_translate
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split


class TMHLoader(Dataset):
    def __init__(self, df, db, mean_embedding=False):
        super().__init__()
        self.df = df
        self.db = db
        self.mean_embedding = mean_embedding

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        hash = self.df.loc[item, "seq_hash"]
        embedding = self.db.get(hash)
        embedding = np.array(embedding).astype(np.float32)
        if np.isnan(embedding).any():
            embedding = np.zeros((1,1024)).astype(np.float32)
        label = self.df.loc[item, "class"]
        if self.mean_embedding:
            embedding = embedding.mean(axis=0)
        return embedding, label

class TMH(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.label_mappings = {
            'Glob_SP': 0,
            'Glob': 1,
            'TM_SP': 2,
            'TM': 3
        }
        self.reverse_label_mappings = {val: key for key, val in self.label_mappings.items()}

    def prepare_data(self):
        root = Path(self.cfg.data_root)
        processed_df_path = root / "processed.pkl"

        if processed_df_path.exists() and not self.cfg.reload_data:
            print("Processed dataset found, loading pickle.")
            df = pd.read_pickle(processed_df_path)
        else:
            print("No processed dataset found, loading from scratch")
            df = self._reload_data(data_root=root)
            print("Done. Dumping to pickle to save time next run")
            df.to_pickle(processed_df_path)
        self.weights = self._calculate_weights(df)

        embeddings = h5py.File(root / "embeddings.h5")
        dataloader = TMHLoader(df, embeddings, self.cfg.mean_embedding)
        val_items = int(len(dataloader) * self.cfg.dataset_val_percentage)
        test_items = int(len(dataloader) * self.cfg.dataset_test_percentage)
        train_items = len(dataloader)-val_items-test_items
        train_set, val_set, test_set = random_split(dataloader, lengths=[train_items, val_items, test_items])

        self.datasets = {
            "train": train_set,
            "val": val_set,
            "test": test_set
        }

    def train_dataloader(self):
        return self._dataloader(mode="train")

    def val_dataloader(self):
        return self._dataloader(mode="val")

    def test_dataloader(self):
        return self._dataloader(mode="test")

    def _dataloader(self, mode):
        return DataLoader(dataset=self.datasets[mode],
                          shuffle=mode == "train",
                          num_workers=self.cfg.num_workers,
                          batch_size=self.cfg.batch_size)

    def _calculate_weights(self, df):
        samples = df["class"].value_counts()
        max_count = samples.max()
        weights = max_count / samples
        return torch.Tensor(weights)


    def _parse_3line(self, path: str, label: str) -> List[SeqRecord]:
        file = open(path)

        def record_for(id: str, seq: str, anno: str) -> SeqRecord:
            # put back together to feed into SeqIO
            # alternative would be to manually construct SeqRecord here
            record = SeqIO.read(StringIO(id + seq), "fasta")
            record.letter_annotations["tmh"] = anno.strip()
            record.annotations["label"] = self.label_mappings[label] # convert str label to int here
            return record

        return [record_for(id, seq, anno) for id, seq, anno in zip(file, file, file)]

    def _as_dict(self, record: SeqRecord) -> dict:
        # we don't currently use the base sequence for training, but it's here for future use
        seq = str(record.seq).strip().replace('X', '*') # dnachisel doesn't like X as stop codons
        base_seq = reverse_translate(seq)
        hash = hashlib.md5(seq.encode("UTF-8")).hexdigest()
        return {
            "protein": record.name,
            "base_seq": base_seq,
            "seq": seq,
            "seq_anno": record.letter_annotations["tmh"],
            "seq_hash": hash,
            "class": record.annotations["label"]
        }

    def _reload_data(self, data_root: Path):
        labels = list(self.label_mappings.keys())
        all_dicts = [self._as_dict(record)
                     for label in labels
                     for record in self._parse_3line(data_root / f"labels/{label}.3line", label=label)]
        return pd.DataFrame(all_dicts)