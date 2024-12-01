from itertools import combinations
import math

import numpy as np
from tqdm import tqdm

# TODO: Maybe save and compute once..
# TODO: Maybe use a datastructure
def find_partitions(elements, group_size, num_groups,progress_bar):
    """
    Recursively partitions `elements` into `num_groups` groups, each of `group_size`.
    Returns a list of unique partitions.
    """
    if num_groups == 1:
        return [list(elements)]
    
    partitions = []
    for group in combinations(elements, group_size):
        remaining_elements = set(elements) - set(group)
        progress_bar.update(1)
        for rest in find_partitions(remaining_elements, group_size, num_groups - 1,progress_bar):
            partitions.append([list(group)] + rest)
    
    return partitions

# define 
# combinatrics
# simplified algo???
# search
#   greedy search 
#   beam search 
#   djsiktsra

# 16 choose 2 = 120

def score(vals, sims):
    groups_of_four = [vals[i*4:(i+1)*4] for i in range(4)]
    scores_of_four = [np.prod()]
    return np.prod(scores_of_four)
    
def djisktra_search(sims):
    vals = tuple(range(16))
    traversed_state_scores[vals] = score(vals, sims)
    
similarity_pairs = [frozenset(item) for item in combinations(range(16),2)]
groups_of_four = [frozenset(item) for item in combinations(range(16),4)]
similarities = np.random.rand(120)
sim_lookup = {pair : similarity for pair,similarity in zip(similarity_pairs,similarities)}

groups_of_four_scores_lookup = {frozenset(group_of_four) : np.prod([sim_lookup[frozenset(pairs)] for pairs in combinations(group_of_four,2)]) for group_of_four in groups_of_four}

import pdb;pdb.set_trace()
# Generate proposed groups of sixteen
proposed_groups_of_sixteen = []
proposed_groups_of_sixteen_scores = []

total_combinations = math.comb(16, 4) * math.comb(12, 4) * math.comb(8, 4) * math.comb(4, 4)

# Create a tqdm progress bar
with tqdm(total=total_combinations, desc="Processing partitions") as progress_bar:
    proposed_groups_of_sixteen = find_partitions(bytearray(range(16)), 4, 4,progress_bar)
        
import pdb;pdb.set_trace()
proposed_groups_of_sixteen_scores = [np.prod([groups_of_four_scores_lookup[frozenset(proposed_group_of_four)] for proposed_group_of_four in proposed_groups_of_four] ) for proposed_groups_of_four in proposed_groups_of_sixteen]
max_idx = np.argmax(proposed_groups_of_sixteen_scores)
best_group = proposed_groups_of_sixteen[max_idx]

# GROUPS OF FOUR

# Could use search instead and maximize the similarity, would be depdent on the intial seed
# could do the search process K times
# I can compute global maxima using the tool above so could figure