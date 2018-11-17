# -*- coding: utf-8 -*-
from __future__ import print_function

import multiprocessing
import random
import time
from itertools import chain

import numpy as np
from networkx import nx


'''
#-----------------------------------------------------------------------------
# part of code was originally forked from https://github.com/thunlp/OpenNE
# modified by Chengbin Hou @ SUSTech 2018
# Email: Chengbin.Hou10@foxmail.com
# ***class BiasedWalker was created by Chengbin Hou
# ***we realize two ways to do ABRW 
#       1) naive sampling (also multi-processor version)
#       2) alias sampling (similar to node2vec)
#-----------------------------------------------------------------------------
'''


def deepwalk_walk_wrapper(class_instance, walk_length, start_node):
    class_instance.deepwalk_walk(walk_length, start_node)


# ===========================================ABRW-weighted-walker============================================
class BiasedWalker:  # ------ our method
    def __init__(self, g, P, workers):
        self.G = g.G  # nx data stcuture
        self.P = P  # biased transition probability; n*n; each row is a pdf for a node
        self.workers = workers
        self.node_size = g.node_size
        self.look_back_list = g.look_back_list
        self.look_up_dict = g.look_up_dict

    # alias sampling for ABRW-------------------------------------------------------------------
    def simulate_walks(self, num_walks, walk_length):
        self.P_G = nx.to_networkx_graph(self.P, create_using=nx.DiGraph())  # create a new nx graph based on ABRW transition prob matrix
        t1 = time.time()
        self.preprocess_transition_probs()  # note: we simply adapt node2vec
        t2 = time.time()
        print('Time for construct alias table: {:.2f}'.format(t2-t1))
        walks = []
        nodes = list(self.P_G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        for i in range(len(walks)):  # use ind to retrive orignal node ID
            for j in range(len(walks[0])):
                walks[i][j] = self.look_back_list[int(walks[i][j])]
        return walks

    def node2vec_walk(self, walk_length, start_node):  # to do...
        G = self.P_G  # more efficient way instead of copy from node2vec
        alias_nodes = self.alias_nodes
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
            else:
                break
        return walk

    def preprocess_transition_probs(self):
        G = self.P_G
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)
        self.alias_nodes = alias_nodes


'''
    #naive sampling for ABRW-------------------------------------------------------------------
    def weighted_walk(self, start_node):
        #
        #Simulate a weighted walk starting from start node.
        #
        G = self.G
        look_up_dict = self.look_up_dict
        look_back_list = self.look_back_list
        node_size = self.node_size
        walk = [start_node]

        while len(walk) < self.walk_length:
            cur_node = walk[-1]        #the last one entry/node
            cur_ind = look_up_dict[cur_node]        #key -> index
            pdf = self.P[cur_ind,:]    #the pdf of node with ind
            #pdf = np.random.randn(18163)+10  #......test multiprocessor
            #pdf = pdf / pdf.sum()            #......test multiprocessor
            #next_ind = int( np.array( nx.utils.random_sequence.discrete_sequence(n=1,distribution=pdf) ) )
            next_ind = np.random.choice(len(pdf), 1, p=pdf)[0]  #faster than nx
            #next_ind = 0                     #......test multiprocessor
            next_node = look_back_list[next_ind]    #index -> key
            walk.append(next_node)
        return walk

    def simulate_walks(self, num_walks, walk_length):
        #
        #Repeatedly simulate weighted walks from each node.
        #
        G = self.G
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.walks = []  #what we all need later as input to skip-gram 
        nodes = list(G.nodes())

        print('Walk iteration:')
        for walk_iter in range(num_walks):
            t1 = time.time()
            random.shuffle(nodes)
            for node in nodes:                              #for single cpu, if # of nodes < 2000 (speed up) or nodes > 20000 (avoid memory error)
                self.walks.append(self.weighted_walk(node)) #for single cpu, if # of nodes < 2000 (speed up) or nodes > 20000 (avoid memory error)
            #pool = multiprocessing.Pool(processes=3)  #use all cpu by defalut or specify processes = xx 
            #self.walks.append(pool.map(self.weighted_walk, nodes))   #ref: https://stackoverflow.com/questions/8533318/multiprocessing-pool-when-to-use-apply-apply-async-or-map
            #pool.close()
            #pool.join()
            t2 = time.time()
            print(str(walk_iter+1), '/', str(num_walks), ' each itr last for: {:.2f}s'.format(t2-t1))
        #self.walks = list(chain.from_iterable(self.walks))  #unlist...[[[x,x],[x,x]]] -> [x,x], [x,x]
        return self.walks
'''


# ===========================================deepWalk-walker============================================
class BasicWalker:
    def __init__(self, G, workers):
        self.G = G.G
        self.node_size = G.get_num_nodes()
        self.look_up_dict = G.look_up_dict

    def deepwalk_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        look_up_dict = self.look_up_dict
        node_size = self.node_size

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
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            # pool = multiprocessing.Pool(processes = 4)
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                # walks.append(pool.apply_async(deepwalk_walk_wrapper, (self, walk_length, node, )))
                walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=node))
            # pool.close()
            # pool.join()
        # print(len(walks))
        return walks


# ===========================================node2vec-walker============================================
class Walker:
    def __init__(self, G, p, q, workers):
        self.G = G.G
        self.p = p
        self.q = q
        self.node_size = G.node_size
        self.look_up_dict = G.look_up_dict

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        look_up_dict = self.look_up_dict
        node_size = self.node_size

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    pos = (prev, cur)
                    next = cur_nbrs[alias_draw(alias_edges[pos][0],
                                               alias_edges[pos][1])]
                    walk.append(next)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
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
        G = self.G

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        look_up_dict = self.look_up_dict
        node_size = self.node_size
        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


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

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
