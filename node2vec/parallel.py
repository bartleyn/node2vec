import random
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
np_where = np.where


FIRST_TRAVEL_KEY = 'first_travel_key'
PROBABILITIES_KEY = 'probabilities'
NEIGHBORS_KEY = 'neighbors'
WEIGHT_KEY = 'weight'
NUM_WALKS_KEY = 'num_walks'
WALK_LENGTH_KEY = 'walk_length'
P_KEY = 'p'
Q_KEY = 'q'


def parallel_generate_walks(d_graph: dict, global_walk_length: int, num_walks: int, cpu_num: int,
                            sampling_strategy: dict = None, num_walks_key: str = None, walk_length_key: str = None,
                            neighbors_key: str = None, probabilities_key: str = None, first_travel_key: str = None,
                            quiet: bool = False) -> list:
    """
    Generates the random walks which will be used as the skip-gram input.

    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get(neighbors_key, None)
                # Skip dead end nodes
                if not walk_options:
                    break

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                else:
                    try:
                        probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
                    except KeyError as e:
                        raise KeyError("{}".format(e)+" walk -1 {} in d_graph {} and walk -2 {} in prob key of -1 {}".format(walk[-1], walk[-1] in d_graph, walk[-2], walk[-2] in d_graph[walk[-1]][probabilities_key]))
                        #raise KeyError(e)
                    try:
                        walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                    except ValueError as e:
                        print(len(probabilities), len(walk_options))
                        raise ValueError(e)

                walk.append(walk_to)

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks

def parallel_precompute_probabilities(d_graph:dict, weight_matrix: csr_matrix, nodes: list,p: int = 1, q: int = 1,
                                          sampling_strategy: dict = None,quiet: bool = False) -> list:
    
        """
        Precomputes transition probabilities for each node.
        """
        first_travel_done = set()
        # do the initial travel in the main func
        num_total_nodes = weight_matrix.shape[0]
        indices = weight_matrix.indices
        indptr = weight_matrix.indptr
        if not quiet:
            pbar = tqdm(total=len(nodes), desc='Computing transition probabilities')
        nodes = [int(x) for x in nodes]
        for source in nodes:
            if not quiet:
                pbar.update(1)
            d_graph[source][NEIGHBORS_KEY] = []
            d_graph[source][NEIGHBORS_KEY] = [int(x) for x in indices[indptr[source]:indptr[source+1]] if x != source]
            # Init probabilities dict for first travel
            if PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][PROBABILITIES_KEY] = dict()
            for current_node in d_graph[source][NEIGHBORS_KEY]:

                if current_node not in d_graph:
                    d_graph[current_node] = {}
                # Init probabilities dict
                if NEIGHBORS_KEY not in d_graph[current_node]:
                    d_graph[current_node][NEIGHBORS_KEY] = [int(x) for x in indices[indptr[current_node]:indptr[current_node+1]] if x != current_node]
                if PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][PROBABILITIES_KEY] = dict()
                
                unnormalized_weights = list()
                first_travel_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for destination in d_graph[current_node][NEIGHBORS_KEY]:

                    p = sampling_strategy[current_node].get(P_KEY,
                                                                 p) if current_node in sampling_strategy else p
                    q = sampling_strategy[current_node].get(Q_KEY,
                                                                 q) if current_node in sampling_strategy else q

                    if destination == source:  # Backwards probability
                        ss_weight = weight_matrix[current_node,destination] * 1 / p
                    elif destination in d_graph[source][NEIGHBORS_KEY]:  # If the neighbor is connected to the source
                        ss_weight = weight_matrix[current_node ,destination]
                    else:
                        ss_weight = weight_matrix[current_node, destination] * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    if current_node not in first_travel_done:
                        first_travel_weights.append(weight_matrix[current_node,destination])



                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][PROBABILITIES_KEY][
                    source] = unnormalized_weights / unnormalized_weights.sum()
                
                if current_node not in first_travel_done:
                    unnormalized_weights = np.array(first_travel_weights)
                    d_graph[current_node][FIRST_TRAVEL_KEY] = unnormalized_weights / unnormalized_weights.sum()
                    first_travel_done.add(current_node)
                
        if not quiet:
            pbar.close()
        return d_graph

