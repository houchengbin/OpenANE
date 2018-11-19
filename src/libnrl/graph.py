"""
commonly used graph APIs based NetworkX;
use g.xxx to access the commonly used APIs offered by us;
use g.G.xxx to access NetworkX APIs;

by Chengbin Hou 2018 <chengbin.hou10@foxmail.com>
"""

import time
import random
import numpy as np
import scipy.sparse as sp
import networkx as nx

class Graph(object):
    def __init__(self):
        self.G = None  #to access NetworkX graph data structure
        self.look_up_dict = {}    #use node ID to find index via g.look_up_dict['0']
        self.look_back_list = []  #use index to find node ID via g.look_back_list[0]
    
    #--------------------------------------------------------------------------------------
    #--------------------commonly used APIs that will modify graph-------------------------
    #--------------------------------------------------------------------------------------
    def node_mapping(self):
        """ node id and index mapping; \n
            based on the order given by networkx G.nodes(); \n
            NB: updating is needed if any node is added/removed; \n
        """
        i = 0 #node index
        self.look_up_dict = {} #init
        self.look_back_list = [] #init
        for node_id in self.G.nodes(): #node id
            self.look_up_dict[node_id] = i
            self.look_back_list.append(node_id)
            i += 1
    
    def read_adjlist(self, path, directed=False):
        """ read adjacency list format graph; \n
            support unweighted and (un)directed graph; \n
            format: see https://networkx.github.io/documentation/stable/reference/readwrite/adjlist.html \n
            NB: not supoort weighted graph \n
        """
        if directed:
            self.G = nx.read_adjlist(path, create_using=nx.DiGraph())
        else:
            self.G = nx.read_adjlist(path, create_using=nx.Graph())
        self.node_mapping() #update node id index mapping

    def read_edgelist(self, path, weighted=False, directed=False):
        """ read edge list format graph; \n
            support (un)weighted and (un)directed graph; \n
            format: see https://networkx.github.io/documentation/stable/reference/readwrite/edgelist.html \n
        """
        if directed:
            self.G = nx.read_edgelist(path, create_using=nx.DiGraph())
        else:
            self.G = nx.read_edgelist(path, create_using=nx.Graph())
        self.node_mapping() #update node id index mapping
    
    def add_edge_weight(self, equal_weight=1.0):
        ''' add weights to networkx graph; \n
            currently only support adding 1.0 to all existing edges; \n
            some NE method may require 'weight' attribute spcified in networkx graph; \n
            to do... support user-specified weights e.g. from file (similar to read_node_attr): node_id1 node_id2 weight \n
            https://networkx.github.io/documentation/stable/reference/generated/networkx.classes.function.set_edge_attributes.html#networkx.classes.function.set_edge_attributes
        '''
        nx.set_edge_attributes(self.G, equal_weight, 'weight') #check the url and use dict to assign diff weights to diff edges
    
    def read_node_attr(self, path):
        """ read node attributes and store as NetworkX graph {'node_id': {'attr': values}} \n
            input file format: node_id1 attr1 attr2 ... attrM \n
                               node_id2 attr1 attr2 ... attrM \n
        """
        with open(path, 'r') as fin:
            for l in fin.readlines():
                vec = l.split()
                self.G.nodes[vec[0]]['attr'] = np.array([float(x) for x in vec[1:]])

    def read_node_label(self, path):
        """ todo... read node labels and store as NetworkX graph {'node_id': {'label': values}} \n
            input file format: node_id1 labels \n
                               node_id2 labels \n
        with open(path, 'r') as fin: \n
            for l in fin.readlines(): \n
                vec = l.split() \n
                self.G.nodes[vec[0]]['label'] = np.array([float(x) for x in vec[1:]]) \n
        """
        pass #to do...

    def remove_edge(self, ratio=0.0):
        """ randomly remove edges/links \n
            ratio: the percentage of edges to be removed \n
            edges_removed: return removed edges, each of which is a pair of nodes \n
        """
        num_edges_removed = int( ratio * self.G.number_of_edges() )
        #random.seed(2018)
        edges_removed = random.sample(self.G.edges(), int(num_edges_removed))
        print('before removing, the # of edges: ', self.G.number_of_edges())
        self.G.remove_edges_from(edges_removed)
        print('after removing, the # of edges: ', self.G.number_of_edges())
        return edges_removed

    def remove_node_attr(self, ratio):
        """ todo... randomly remove node attributes; \n
        """
        pass #to do...

    def remove_node(self, ratio):
        """ todo... randomly remove nodes; \n
            #self.node_mapping() #update node id index mapping is needed \n
        """
        pass #to do...
    
    #------------------------------------------------------------------------------------------
    #--------------------commonly used APIs that will not modify graph-------------------------
    #------------------------------------------------------------------------------------------
    def get_adj_mat(self, is_sparse=True):
        """ return adjacency matrix; \n
            use 'csr' format for sparse matrix \n
        """
        if is_sparse:
            return nx.to_scipy_sparse_matrix(self.G, nodelist=self.look_back_list, format='csr', dtype='float64')
        else:
            return nx.to_numpy_matrix(self.G, nodelist=self.look_back_list, dtype='float64')

    def get_attr_mat(self, is_sparse=True):
        """ return attribute matrix; \n
            use 'csr' format for sparse matrix \n
        """
        attr_dense_narray = np.vstack([self.G.nodes[self.look_back_list[i]]['attr'] for i in range(self.get_num_nodes())])
        if is_sparse:
            return sp.csr_matrix(attr_dense_narray, dtype='float64')
        else:
            return np.matrix(attr_dense_narray, dtype='float64')

    def get_num_nodes(self):
        """ return the number of nodes """
        return nx.number_of_nodes(self.G)

    def get_num_edges(self):
        """ return the number of edges """
        return nx.number_of_edges(self.G)

    def get_density(self):
        """ return the density of a graph """
        return nx.density(self.G)

    def get_num_isolates(self):
        """ return the number of isolated nodes """
        return len(list(nx.isolates(self.G)))

    def get_isdirected(self):
        """ return True if it is directed graph """
        return nx.is_directed(self.G)

    def get_isweighted(self):
        """ return True if it is weighted graph """
        return nx.is_weighted(self.G)
    
    def get_neighbors(self, node):
        """ return neighbors connected to a node """
        return list(nx.neighbors(self.G, node))

    def get_common_neighbors(self, node1, node2):
        """ return common neighbors of two nodes """
        return list(nx.common_neighbors(self.G, node1, node2))

    def get_centrality(self, centrality_type='degree'):
        """ todo... return specified type of centrality \n
            see https://networkx.github.io/documentation/stable/reference/algorithms/centrality.html \n
        """ 
        pass #to do...

