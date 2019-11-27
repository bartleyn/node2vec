import networkx as nx
from node2vec import Node2Vec
import numpy as np


# FILES
EMBEDDING_FILENAME = './embeddings.emb'
EMBEDDING_MODEL_FILENAME = './embeddings.model'
temp_folder = './tmp'

# Create a graph
graph = nx.fast_gnp_random_graph(n=100, p=0.5)

#weight_mat = nx.adjacency_matrix(graph, weight=True)
#indices = weight_mat.indices
#indptr = weight_mat.indptr
#print(list(graph.neighbors(1)))
#print([x for x in indices[indptr[1]:indptr[2]]])
#raise Exception


# Precompute probabilities and generate walks
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder=temp_folder)

## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
# Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
#node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")

# Embed
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
print(model.wv.most_similar('2'))  # Output node names are always strings

node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=1, temp_folder=temp_folder)
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
print(model.wv.most_similar('2'))  # Output node names are always strings



# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
model.save(EMBEDDING_MODEL_FILENAME)
