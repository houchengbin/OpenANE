"""
ANE method: Attributed Biased Random Walks;

by Chengbin Hou & Zeyu Dong 2018
"""

import time
import warnings

import numpy as np
from gensim.models import Word2Vec
from scipy import sparse

from . import walker
from .utils import pairwise_similarity, row_as_probdist

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


class ABRW(object):
    def __init__(self, graph, dim, alpha, topk, number_walks, walk_length, **kwargs):
        self.g = graph
        self.dim = dim
        self.alpha = float(alpha)
        self.topk = int(topk)
        self.number_walks = number_walks
        self.walk_length = walk_length

        # obtain biased transition mat -----------
        self.T = self.get_biased_transition_mat(A=self.g.get_adj_mat(), X=self.g.get_attr_mat())

        # aim to generate a sequences of walks/sentences
        # apply weighted random walks on the reconstructed network based on biased transition mat
        kwargs["workers"] = kwargs.get("workers", 8)
        weighted_walker = walker.WeightedWalker(node_id_map=self.g.look_back_list, transition_mat=self.T, workers=kwargs["workers"])  # instance weighted walker
        sentences = weighted_walker.simulate_walks(num_walks=self.number_walks, walk_length=self.walk_length)

        # feed the walks/sentences into Word2Vec Skip-Gram model for traning node embeddings
        kwargs["sentences"] = sentences
        kwargs["size"] = self.dim
        kwargs["sg"] = 1  # use skip-gram; but see deepwalk which uses 'hs' = 1
        kwargs["window"] = kwargs.get("window", 10)
        kwargs["min_count"] = kwargs.get("min_count", 0)  # drop words/nodes if below the min_count freq; set to 0 to get all node embs
        print("Learning node embeddings......")
        word2vec = Word2Vec(**kwargs)

        # save emb as a dict
        self.vectors = {}
        for word in self.g.G.nodes():
            self.vectors[word] = word2vec.wv[word]
        del word2vec

    def get_biased_transition_mat(self, A, X):
        '''
        given: A and X --> T_A and T_X
        research question: how to combine A and X in a more principled way
        genral idea: Attribute Biased Random Walk
        i.e. a walker based on a mixed transition matrix by P=alpha*T_A + (1-alpha)*T_X
        result: ABRW-trainsition matrix; T
        '''
        print("obtaining biased transition matrix where each row sums up to 1.0...")

        preserve_zeros = False
        T_A = row_as_probdist(A, preserve_zeros)  # norm adj/struc info mat; for isolated node, return all-zeros row or all-1/m row
        print('Preserve zero rows of the adj matrix: ', preserve_zeros)

        t1 = time.time()
        X_sim = pairwise_similarity(X)  # attr similarity mat; X_sim is a square mat, but X is not

        t2 = time.time()
        print(f'keep the top {self.topk} attribute similar nodes w.r.t. a node')
        cutoff = np.partition(X_sim, -self.topk, axis=1)[:, -self.topk:].min(axis=1)
        X_sim[(X_sim < cutoff)] = 0  # improve both accuracy and efficiency
        X_sim = sparse.csr_matrix(X_sim)

        t3 = time.time()
        T_X = row_as_probdist(X_sim)

        t4 = time.time()
        print(f'attr sim cal time: {(t2-t1):.2f}s; topk sparse ops time: {(t3-t2):.2f}s; row norm time: {(t4-t3):.2f}s')
        del A, X, X_sim

        # =====================================information fusion via transition matrices========================================
        print('------alpha for P = alpha * T_A + (1-alpha) * T_X------: ', self.alpha)
        n = self.g.get_num_nodes()
        alp = np.array(n * [self.alpha])  # for vectorized computation
        alp[~np.asarray(T_A.sum(axis=1) != 0).ravel()] = 0
        T = sparse.diags(alp).dot(T_A) + sparse.diags(1 - alp).dot(T_X)  # sparse version
        t5 = time.time()
        print(f'ABRW biased transition matrix processing time: {(t5-t4):.2f}s')
        return T

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.dim))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()
