"""
(weighted) random walks for walk-based NE methods:
DeepWalk, Node2Vec, TriDNR, and ABRW;
alias sampling; walks by multiprocessors; etc.

by Chengbin Hou & Zeyu Dong 2018
"""

import functools
import multiprocessing
import random
import time

import numpy as np
from networkx import nx


# ===========================================ABRW-weighted-walker============================================
class WeightedWalker:
    ''' Weighted Walker for Attributed Biased Randomw Walks (ABRW) method
    '''

    def __init__(self, node_id_map, transition_mat, workers):
        self.look_back_list = node_id_map
        self.T = transition_mat
        self.workers = workers
        self.rec_G = nx.to_networkx_graph(self.T, create_using=nx.DiGraph())  # reconstructed "directed" "weighted" graph based on transition matrix

    # alias sampling for ABRW-------------------------
    def simulate_walks(self, num_walks, walk_length):
        global P_G
        P_G = self.rec_G
        
        t1 = time.time()
        self.preprocess_transition_probs(weighted_G=self.rec_G)  # construct alias table; adapted from node2vec
        t2 = time.time()

        global alias_nodes
        alias_nodes = self.alias_nodes
        print(f'Time for construct alias table: {(t2-t1):.2f}')
        
        walks = []
        nodes = list(self.rec_G.nodes())
        pool = multiprocessing.Pool(self.workers)
        for walk_iter in range(num_walks):
            t1 = time.time()
            random.shuffle(nodes)
            walks += pool.map(functools.partial(node2vec_walk, walk_length=walk_length), nodes)
            t2 = time.time()
            print(f'Walk iteration: {walk_iter+1}/{num_walks}; time cost: {(t2-t1):.2f}')
        pool.close()
        pool.join()
        del alias_nodes, P_G

        for i in range(len(walks)):  # use ind to retrive original node ID
            for j in range(len(walks[0])):
                walks[i][j] = self.look_back_list[int(walks[i][j])]
        return walks

    def preprocess_transition_probs(self, weighted_G):
        ''' reconstructed G mush be weighted; \n
            return a dict of alias table for each node
        '''
        G = weighted_G
        alias_nodes = {}
        nodes = G.nodes()

        pool = multiprocessing.Pool(self.workers)
        alias_nodes = dict(zip(nodes, pool.map(get_alias_node, nodes)))
        pool.close()
        pool.join()

        self.alias_nodes = alias_nodes


def node2vec_walk(start_node, walk_length):  # to do...
    global P_G  # more efficient way instead of copy from node2vec
    global alias_nodes
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(P_G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
        else:
            break
    return walk


def get_alias_node(node):
    global P_G
    probs = [P_G[node][nbr]['weight'] for nbr in P_G.neighbors(node)]
    return alias_setup(probs)



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
