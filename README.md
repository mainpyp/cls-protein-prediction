# cls-protein-prediction

## Usage

1. Download the weights from [Google Drive](https://drive.google.com/drive/folders/1jaRYaAKIe5YwUb7xJKqSsAhZA1A6v7oQ?usp=sharing) and place in a folder called `models` in the root
2. Download tmh dataset from moodle or [Nextcloud](https://nextcloud.in.tum.de/index.php/s/6Hbq9QEFbgKQ5m7) (Password protected)
3. Download FASTA file for evaluation from the same source
4. Install dependencies
```
pip install -r requirements.txt
```
6. Run main script, supplying at least the following arguments
```
python main.py 
    --model_type <one of: ["CNN", "MLP", "CAIT"]>
    --emb <path to embeddings.h5> 
    --fasta <path to FASTA file>
```

## Relevant Links

- [Pytorch Transformers for Machine Translation](https://www.youtube.com/watch?v=M6adRGJe5cQ)
- [Pytorch Transformers from Scratch (Attention is all you need)](https://www.youtube.com/watch?v=U0s0f995w14)

