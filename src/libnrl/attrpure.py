"""
NE method: use only attribute information (AttrPure)

by Chengbin Hou 2018
"""

import time

import networkx as nx
import numpy as np

from .utils import dim_reduction

class ATTRPURE(object):

    def __init__(self, graph, dim, mode):
        self.g = graph
        self.dim = dim
        self.mode = mode
        
        print("Learning representation...")
        self.vectors = {}
        embeddings = self.train()
        for key, ind in self.g.look_up_dict.items():
            self.vectors[key] = embeddings[ind]

    def train(self):
        X = self.g.get_attr_mat().todense()
        X_compressed = None
        if self.mode == 'pca':
            X_compressed = dim_reduction(X, dim=self.dim, method='pca')
        elif self.mode == 'svd':
            X_compressed = dim_reduction(X, dim=self.dim, method='svd')
        else:
            print('unknown dim reduction technique...')
        return X_compressed


    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.dim))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()     
