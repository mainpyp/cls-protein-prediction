import argparse
import hashlib



################# ArgumentParser #################

parser = argparse.ArgumentParser(description='Class Attention Model')
parser.add_argument('--fasta',type=str,
                    help='path_to_fasta')
parser.add_argument('--emb',type=str,
                    help='path_to_embeddings')
parser.add_argument('--output',type=str,
                    help='path_to_output')
args = parser.parse_args()

fasta_path = parser.fasta + ".fasta"
h5_path = parser.emb + ".h5"
output_path = parser.output if parser.output + "/output.tsv" is not None else "./output.tsv"



################# Load Data #################





################# Test Model #################





################# Write Output #################

# <prot_id>\t<label>\t<confidence_score>
# prot_id as given in h5 or fasta
# label TM, TM+SP, G, G+SP
# confidence score between 0 and 1


