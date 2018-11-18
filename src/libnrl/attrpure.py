# -*- coding: utf-8 -*-
import time

import networkx as nx
import numpy as np

from .utils import dim_reduction


'''
#-----------------------------------------------------------------------------
# author: Chengbin Hou 2018
# Email: Chengbin.Hou10@foxmail.com
#-----------------------------------------------------------------------------
'''

class ATTRPURE(object):

    def __init__(self, graph, dim):
        self.g = graph
        self.dim = dim
        
        print("Learning representation...")
        self.vectors = {}
        embeddings = self.train()
        for key, ind in self.g.look_up_dict.items():
            self.vectors[key] = embeddings[ind]

    def train(self):
        X = self.g.get_attr_mat()
        X_compressed = dim_reduction(X, dim=self.dim, method='svd')  #svd or pca for dim reduction
        return X_compressed    #n*dim matrix, each row corresponding to node ID stored in graph.look_back_list


    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.dim))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()     
