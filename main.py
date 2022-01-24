import hashlib
from io import StringIO
from typing import List

import h5py
import numpy as np
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from common.cait_models import CaiT
from common.models import *
from common.module import CaitModule
################# ArgumentParser #################
from train import load_cfg

cfg = load_cfg()


################# Mappings for labels #################
label_mappings = {"Glob_SP": 0, "Glob": 1, "TM_SP": 2, "TM": 3}
reverse_label_mappings = {val: key for key, val in label_mappings.items()}


################# Read Data #################
def read_fasta(path: str) -> List[SeqRecord]:
    file = open(path)

    def record_for(id: str, seq: str) -> SeqRecord:
        # put back together to feed into SeqIO
        # alternative would be to manually construct SeqRecord here
        record = SeqIO.read(StringIO(id + seq), "fasta")
        return record

    return [record_for(id, seq) for id, seq in zip(file, file)]


def as_dict(record: SeqRecord) -> dict:
    # we don't currently use the base sequence for training, but it's here for future use
    seq = (
        str(record.seq).strip().replace("X", "*")
    )  # dnachisel doesn't like X as stop codons
    hash = hashlib.md5(seq.encode("UTF-8")).hexdigest()
    return {"protein": record.name, "seq": seq, "seq_hash": hash}


def complete_dicts(emb_path: str, dicts: List[dict]) -> List[dict]:
    h5 = h5py.File(emb_path)
    for each_dict in dicts:
        hash = each_dict["seq_hash"]

        embedding = h5.get(hash)
        embedding = np.array(embedding).astype(np.float32)
        if np.isnan(embedding).any():
            embedding = np.zeros((1, 1024)).astype(np.float32)

        each_dict["embedding"] = embedding
    return dicts


def read_data(path_fasta: str, path_h5: str) -> List[dict]:
    records = read_fasta(path_fasta)
    all_dicts = [as_dict(record) for record in records]
    all_dicts = complete_dicts(path_h5, all_dicts)
    return all_dicts


################# Load Model #################
def load_model(model_type="CAIT") -> CaitModule:
    if model_type == "CAIT":
        model = CaiT(cfg)
    elif model_type == "MLP":
        model = MLP(cfg)
    elif model_type == "CNN":
        model = CNN(cfg)
    module = CaitModule(cfg, model)
    ckpt = torch.load(f"models/{model_type}.ckpt", map_location="cpu")
    module.load_state_dict(ckpt["state_dict"])
    module = module.eval()
    return module


################# Predict Data #################
def predict(data: List[dict], module: nn.Module) -> dict:
    with torch.no_grad():
        result_predicted = dict()
        for each_dict in data:
            emb_as_tensor = torch.Tensor(each_dict["embedding"]).unsqueeze(dim=0)
            prediction_output, attn = module.model.forward(emb_as_tensor)
            probability_output = F.softmax(prediction_output)
            confidence = torch.max(probability_output)
            predicted = torch.argmax(probability_output)
            assert predicted in range(4), "Something went wrong with the prediction"
            string_label = reverse_label_mappings[int(predicted)]
            result_predicted[each_dict["protein"]] = [string_label, confidence]
    return result_predicted


################# Write Output #################
def parse_output(results: dict, output_path: str) -> None:
    with open(output_path, "w") as file:
        for protein, prediction in results.items():
            file.write(f"{protein}\t{prediction[0]}\t{prediction[1]:.2f}\n")


if __name__ == "__main__":
    prediction_data = read_data(path_fasta=cfg.fasta, path_h5=cfg.emb)
    model = load_model(cfg.model_type)
    output = predict(prediction_data, model)
    parse_output(output, output_path=cfg.output)
