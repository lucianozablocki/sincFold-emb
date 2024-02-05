import torch as tr
import numpy as np
from scipy.spatial.distance import cdist
import sys
from numpy import ma

embeddings_file_path = sys.argv[1]
output_file_path = sys.argv[2]
print(f"reading file at {embeddings_file_path}")
id_to_embedding_seq = tr.load(embeddings_file_path)

max_L = -1
for tensor in id_to_embedding_seq.values():
    L, d = tensor.shape
    max_L = max(max_L, L)
print(f"max sequence length is: {max_L}")

# put e
n = len(id_to_embedding_seq)
print(f"there are {n} sequences in the dataset")
all_embedding_seq = np.zeros((n, max_L, d))
for i, embedding_seq in enumerate(id_to_embedding_seq.values()):
    # all_embedding_seq[i] = np.pad(embedding_seq, ((0, max_L - embedding_seq.shape[0]),(0,0)))
    pass
print(f"finished creating matrix with all the embeddings in it, of size {all_embedding_seq.shape}")

all_embeding_seq = ma.masked_array(all_embedding_seq, mask=(all_embedding_seq==0))
print(f"finished masking matrix")

average_embedding = np.mean(all_embedding_seq, axis=1)
print(f"finished calculating averages, matrix is now of size {average_embedding.shape}")

distance_matrix = cdist(average_embedding, average_embedding, 'euclidean')
print(f"finished calculating distance matrix with size of {distance_matrix.shape}")

np.savetxt(output_path, distance_matrix, delimiter=',')
print(f"finished saving distance matrix to disk")
