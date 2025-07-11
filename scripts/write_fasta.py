import pdb
import pandas as pd
from tqdm import tqdm
baseline_path = "/home/qcx679/hantang/UAAG/data/DMS/full_benchmark/DN7A_SACS2_Tsuboyama_2023_1JIC.csv"
pdb_path = "/home/qcx679/hantang/UAAG/data/DMS/pdb/DN7A_SACS2.pdb"
# write the fasta file
df = pd.read_csv(baseline_path)
with open("DN7A_SACS2.fasta", "w") as f:
    for index, row in tqdm(df.iterrows()):
        f.write(">%s\n" % row["mutant"])
        f.write("%s\n" % row["mutated_sequence"])
f.close()
# save the fasta
# from Bio import SeqIO
# from Bio.Seq import Seq
# from Bio.SeqRecord import SeqRecord
# from Bio.Alphabet import IUPAC
