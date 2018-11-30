"""
NE method: naively combine AttrPure and DeepWalk (AttrComb)

by Chengbin Hou 2018
"""

import numpy as np

from . import node2vec
from .utils import dim_reduction


class ATTRCOMB(object):
    def __init__(self, graph, dim, comb_method='concat', comb_with='deepWalk', number_walks=10, walk_length=80, window=10, workers=8):
        self.g = graph
        self.dim = dim
        self.number_walks = number_walks
        self.walk_length = walk_length
        self.window = window
        self.workers = workers

        print("Learning representation...")
        self.vectors = {}

        print('attr naively combined method ', comb_method, '=====================')
        if comb_method == 'concat':
            print('comb_method == concat by default; dim/2 from attr and dim/2 from nrl.............')
            attr_embeddings = self.train_attr(dim=int(self.dim/2))
            nrl_embeddings = self.train_nrl(dim=int(self.dim/2), comb_with='deepWalk')
            embeddings = np.concatenate((attr_embeddings, nrl_embeddings), axis=1)
            print('shape of embeddings', embeddings.shape)

        elif comb_method == 'elementwise-mean':
            print('comb_method == elementwise-mean.............')
            attr_embeddings = self.train_attr(dim=self.dim)
            nrl_embeddings = self.train_nrl(dim=self.dim, comb_with='deepWalk')  # we may try deepWalk, node2vec, line and etc...
            embeddings = np.add(attr_embeddings, nrl_embeddings)/2.0
            print('shape of embeddings', embeddings.shape)

        elif comb_method == 'elementwise-max':
            print('comb_method == elementwise-max.............')
            attr_embeddings = self.train_attr(dim=self.dim)
            nrl_embeddings = self.train_nrl(dim=self.dim, comb_with='deepWalk')  # we may try deepWalk, node2vec, line and etc...
            embeddings = np.zeros(shape=(attr_embeddings.shape[0], attr_embeddings.shape[1]))
            for i in range(attr_embeddings.shape[0]):  # size(attr_embeddings) = size(nrl_embeddings)
                for j in range(attr_embeddings.shape[1]):
                    if attr_embeddings[i][j] > nrl_embeddings[i][j]:
                        embeddings[i][j] = attr_embeddings[i][j]
                    else:
                        embeddings[i][j] = nrl_embeddings[i][j]
            print('shape of embeddings', embeddings.shape)

        else:
            print('error, no comb_method was found....')
            exit(0)

        for key, ind in self.g.look_up_dict.items():
            self.vectors[key] = embeddings[ind]

    def train_attr(self, dim):
        X = self.g.get_attr_mat()
        X_compressed = dim_reduction(X, dim=dim, method='svd')  # svd or pca for dim reduction
        print('X_compressed shape: ', X_compressed.shape)
        return np.array(X_compressed)  # n*dim matrix, each row corresponding to node ID stored in graph.look_back_list

    def train_nrl(self, dim, comb_with):
        print('attr naively combined with ', comb_with, '=====================')
        if comb_with == 'deepWalk':
            model = node2vec.Node2vec(graph=self.g, dim=dim, path_length=self.walk_length,  # do not use self.dim here
                                      num_paths=self.number_walks, workers=self.workers, window=self.window, dw=True)
            nrl_embeddings = []
            for key in self.g.look_back_list:
                nrl_embeddings.append(model.vectors[key])
            return np.array(nrl_embeddings)

        elif comb_with == 'node2vec':  # to do... the parameters
            model = node2vec.Node2vec(graph=self.g, path_length=80, num_paths=self.number_walks,
                                      dim=dim, workers=4, p=0.8, q=0.8, window=10)
            nrl_embeddings = []
            for key in self.g.look_back_list:
                nrl_embeddings.append(model.vectors[key])
            return np.array(nrl_embeddings)

        else:
            print('error, no comb_with was found....')
            print('to do.... line, grarep, and etc...')
            exit(0)

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.dim))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
