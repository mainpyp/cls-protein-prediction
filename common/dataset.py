import hashlib
from io import StringIO
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from dnachisel.biotools import reverse_translate
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


class TMHLoader(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.df.loc[item, "embedding"], self.df.loc[item, "class"]

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
            print("No processed dataset found, loading from scratch...")
            df = self._reload_data(data_root=root)
            df.to_pickle(processed_df_path)

        dataloader = TMHLoader(df)
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

    def _parse_3line(self, path: str, label: str) -> List[SeqRecord]:
        file = open(path)

        def record_for(id: str, seq: str, anno: str) -> SeqRecord:
            amino_seq = seq.strip().replace('X', '*')
            base_seq = reverse_translate(amino_seq)
            # put back together to feed into SeqIO
            # alternative would be to manually construct SeqRecord here
            record = SeqIO.read(StringIO(id + base_seq), "fasta")
            record.annotations["tmh"] = anno.strip()
            record.annotations["label"] = self.label_mappings[label]
            assert record.translate().seq == amino_seq  # sanity check
            return record

        return [record_for(id, seq, anno) for id, seq, anno in zip(file, file, file)]

    def _as_dict(self, record: SeqRecord, embeddings_file) -> dict:
        seq = str(record.translate().seq)
        hash = hashlib.md5(seq.encode("UTF-8")).hexdigest()
        embedding = embeddings_file.get(hash)
        mean_embedding = np.mean(embedding, axis=0) if embedding is not None else np.zeros(1024)
        return {
            "protein": record.name,
            "base_seq": str(record.seq),
            "seq": seq,
            "seq_anno": record.annotations["tmh"],
            "seq_hash": hash,
            "embedding": mean_embedding.astype(np.float32),
            "class": record.annotations["label"]
        }

    def _reload_data(self, data_root: Path):
        embeddings = h5py.File(data_root / "embeddings.h5")
        labels = list(self.label_mappings.keys())
        all_dicts = [self._as_dict(record, embeddings)
                     for label in tqdm(labels)
                     for record in tqdm(self._parse_3line(data_root / f"labels/{label}.3line", label=label))]
        return pd.DataFrame(all_dicts)