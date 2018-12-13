"""
(weighted) random walks for walk-based NE methods:
DeepWalk, Node2Vec, TriDNR, and ABRW;
alias sampling; walks by multiprocessors; etc.

by Chengbin Hou & Zeyu Dong 2018
"""

import random
import time

import numpy as np
from networkx import nx
from scipy import sparse

# ===========================================ABRW-weighted-walker============================================
class WeightedWalker:
    ''' Weighted Walker for Attributed Biased Randomw Walks (ABRW) method
    '''

    def __init__(self, node_id_map, transition_mat, workers):
        assert sparse.isspmatrix_csr(transition_mat)  # currently support csr format
        self.T = transition_mat
        self.look_back_list = node_id_map
        self.workers = workers
        
    # alias sampling for ABRW-------------------------
    def simulate_walks(self, num_walks, walk_length):
        t1 = time.time()
        print('333')
        self.preprocess_transition_probs()  # construct alias table; adapted from node2vec
        print('444')
        t2 = time.time()
        print(f'Time for construct alias table: {(t2-t1):.2f}')

        walks = []
        nodes = [i for i in range(self.T.shape[0])]
        for walk_iter in range(num_walks):
            t1 = time.time()
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.weighted_walk(walk_length=walk_length, start_node=node))
            t2 = time.time()
            print(f'Walk iteration: {walk_iter+1}/{num_walks}; time cost: {(t2-t1):.2f}')

        for i in range(len(walks)):  # use ind to retrive original node ID
            for j in range(len(walks[0])):
                walks[i][j] = self.look_back_list[int(walks[i][j])]
        return walks

    def weighted_walk(self, walk_length, start_node):  # more efficient way instead of copy from node2vec
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]  # the last node
            cur_nbrs = self.T.getrow(cur).nonzero()[1]
            if len(cur_nbrs) > 0:  # if non-isolated node
                walk.append( cur_nbrs[ alias_draw(self.alias_nodes[cur][0], self.alias_nodes[cur][1]) ] )  # alias sampling in O(1) time to get the index of
            else:  # if isolated node                                                                  # 1) randomly choose a nbr; 2) judge if use nbr or its alias
                break
        return walk

    def preprocess_transition_probs(self):
        T = self.T
        alias_nodes = {}
        # alias_nodes = [alias_setup(T.getrow(i).data) for i in range(T.shape[0])]
        
        for i in range(T.shape[0]):
            sparse_row = T.getrow(i)
            # r_index, c_index = sparse_row.nonzero()
            # c_index = sparse_row.nonzero()[1]
            c_value = sparse_row.data
            # assert np.sum(c_value) == 1.0  # error if not prob dist
            # print(np.sum(c_value))
            alias_node = alias_setup(c_value)  # where array0 gives alias node indexes; array1 gives its prob
            # alias_index = [c_index[j] for j in alias_prob_index] # use alias index to find matrix index; and replace
            # alias_nodes[i] = (alias_index, alias_prob)
            alias_nodes[i] = alias_node
        
        self.alias_nodes = alias_nodes


def deepwalk_walk_wrapper(class_instance, walk_length, start_node):
    class_instance.deepwalk_walk(walk_length, start_node)

# ===========================================deepWalk-walker============================================


class BasicWalker:
    def __init__(self, g, workers):
        self.g = g
        self.node_size = g.get_num_nodes()
        self.look_up_dict = g.look_up_dict

    def deepwalk_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.g.G

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.g.G
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(num_walks):
            t1 = time.time()
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=node))
            t2 = time.time()
            print(f'Walk iteration: {walk_iter+1}/{num_walks}; time cost: {(t2-t1):.2f}')
        return walks


# ===========================================node2vec-walker============================================
class Walker:
    def __init__(self, g, p, q, workers):
        self.g = g
        self.p = p
        self.q = q

        if self.g.get_isweighted():
            # print('is weighted graph: ', self.g.get_isweighted())
            pass
        else:  # otherwise, add equal weights 1.0 to all existing edges
            # print('is weighted graph: ', self.g.get_isweighted())
            self.g.add_edge_weight(equal_weight=1.0)  # add 'weight' to networkx graph

        self.node_size = g.get_num_nodes()
        self.look_up_dict = g.look_up_dict

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.g.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.g.G
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(num_walks):
            t1 = time.time()
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
            t2 = time.time()
            print(f'Walk iteration: {walk_iter+1}/{num_walks}; time cost: {(t2-t1):.2f}')
        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.g.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in G.neighbors(dst):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.g.G
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in G.neighbors(node)]  # pick prob of neighbors with non-zero weight
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        if self.g.get_isdirected():
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:  # if undirected, duplicate the reverse direction; otherwise may get key error
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges


# ========================================= utils: alias sampling method ====================================================
def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:  # it is all about use large prob to compensate small prob untill reach the average
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q  # the values in J are indexes; it is possible to have repeated indexes if that that index have large prob to compensate others


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))  # randomly choose a nbr (an index)
    if np.random.rand() < q[kk]:  # use alias table to choose
        return kk  # either that nbr node (an index)
    else:
        return J[kk]  # or the nbr's alias node (an index)
