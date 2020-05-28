"""
ANE method: Adap-ANE: Adaptive Attributed Network Embedding
            based on previous Attributed Biased Random Walks https://arxiv.org/abs/1811.11728v2

by Chengbin Hou & Zeyu Dong 2018
"""

import time
import warnings

import numpy as np
from gensim.models import Word2Vec
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import psutil

from . import walker
from .utils import pairwise_similarity, row_as_probdist


warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def deg2beta_mapping(deg_list, alpha=np.e):
    ''' 
    ***adaptive beta*** 
    based on node degree for balancing structure info and attribute info
    map node degree [0:+inf) -> beta [1:0]
    input:  deg_list: a scalar or list
            alpha: default e; [0, +inf) but we suggest trying 0.5, 1, e, 10, 100, ...
    output: beta_list: a scalar or list
    '''
    base_list = (1.0 + np.power(deg_list, alpha))
    beta_list = np.power(base_list, -1/alpha)  # characteristic curve of adaptive beta
    # print('deg_list', deg_list[:50])
    # print('beta_list', np.around(beta_list, decimals=3)[:50])
    return beta_list


class ABRW(object):
    def __init__(self, graph, dim, topk, beta, beta_mode, alpha, number_walks, walk_length, **kwargs):
        self.g = graph
        self.dim = dim
        self.topk = int(topk)
        self.beta = float(beta)
        self.beta_mode = int(beta_mode)
        self.alpha = float(alpha)
        self.number_walks = number_walks
        self.walk_length = walk_length

        # obtain biased transition mat -----------
        self.T = self.get_biased_transition_mat(A=self.g.get_adj_mat(dense_output=False), X=self.g.get_attr_mat(dense_output=False))

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
        our idea: T = (1-beta)*T_A + beta*T_X
        mode 1: fixed beta
        mode 2: adaptive beta baed on average degree
        mode 3: adaptive beta based on each node degree 
        '''
        print("obtaining biased transition matrix where each row sums up to 1.0...")

        # norm adj mat; For isolated node, return all-zeros row, so that T_A is not a strict transition matrix
        # preserve_all_zero_row=False gives similar result, but is less efficient
        t0 = time.time()
        T_A = row_as_probdist(A, dense_output=False, preserve_all_zero_row=True)    # **sparse mat**
        T_X = None
        
        n = self.g.get_num_nodes()
        free_memory = psutil.virtual_memory().available
        print('free_memory ', free_memory)
        # n*n*8 is the bytes required by pairwise similarity matrix; 2e9 = 2GB ROM remained for safety reason
        # if your computer have 200G memory, there should be no problem for graph with 100k nodes
        # this naive implementation is **faster** than BallTree implementation, thanks to numpy
        if False:   # X_sim[n,n] dense + A[n,n] if dense + X[n,5000] if dense with max 5000 feats + 2e9 for safety
            print('naive implementation + intro-select ')
            t1 = time.time()
            X_sim = pairwise_similarity(X.todense())
            # sparse operator; reduce time and space complexity & remove less useful dissimilar nodes
            t2 = time.time()
            print(f'keep the top {self.topk} attribute similar nodes w.r.t. a node')
            cutoff = np.partition(X_sim, -self.topk, axis=1)[:, -self.topk:].min(axis=1).reshape(-1,1) # introselect average speed O(1); see link below
            X_sim[(X_sim < cutoff)] = 0                                     # https://docs.scipy.org/doc/numpy/reference/generated/numpy.partition.html
            X_sim = sparse.csr_matrix(X_sim)
            X_sim.setdiag(0)
            # norm attr mat; note: T_X mush be a strict transition matrix, thanks to the strict transition matrix of X_sim
            t3 = time.time()
            T_X = row_as_probdist(X_sim, dense_output=False, preserve_all_zero_row=False) # **sparse mat**
            t4 = time.time()
            print(f'attr sim cal time: {(t2-t1):.2f}s; topk sparse ops time: {(t3-t2):.2f}s')
            print(f'adj row norm time: {(t1-t0):.2f}s; attr row norm time: {(t4-t3):.2f}s')
            print('all naive implementation time: ', t4-t1)
            del A, X, X_sim, cutoff
        # a scalable w.r.t. both time and space
        # but might be slightly slower when n is small e.g. n<100k
        # BallTree time complexity O( nlong(n) )
        else:
            print('BallTree implementation + multiprocessor query')
            t1 = time.time()
            X = normalize(X.todense(), norm='l2', axis=1)
            t2 = time.time()
            print('normalize time: ',t2-t1)
            # after normalization -> Euclidean distance = cosine distance (inverse of cosine similarity)
            neigh = NearestNeighbors(n_neighbors=self.topk, algorithm='ball_tree', leaf_size=40, metric='minkowski', p=2, n_jobs=-1)
            neigh.fit(X)
            t3 = time.time()
            print('BallTree time: ',t3-t2)
            dist, ind = neigh.kneighbors(X[:]) # Euclidean dist, indices
            # print('dist',dist)
            # print('ind',ind)
            t4 = time.time()
            print('query time: ',t4-t3)
            sim = 1-np.multiply(dist, dist)/2  # cosine distance -> cosine similarity
            # print('sim: ',sim)
            t5 = time.time()
            print('cosine distance -> cosine similarity time: ',t5-t4)
            row = []
            col = []
            data = []
            for i in range(n):
                row.extend( [i]* self.topk )
                col.extend( ind[i] )
                data.extend( sim[i] )
            t6 = time.time()
            print('sparse matrix data & ind construction for loop time: ',t6-t5)
            zero_row_ind = np.where(~X.any(axis=1))[0]
            # print('zero_row_ind',zero_row_ind)
            X_sim = sparse.csc_matrix((data, (row, col)), shape=(n, n))
            for col in zero_row_ind:
                X_sim.data[X_sim.indptr[col]:X_sim.indptr[col+1]] = 0
            X_sim = sparse.csr_matrix(X_sim)
            for row in zero_row_ind:
                X_sim.data[X_sim.indptr[row]:X_sim.indptr[row+1]] = 0
            X_sim.setdiag(0)
            X_sim.eliminate_zeros()
            t7 = time.time()
            # print(X_sim.todense())
            print('sparse.csr_matrix time:',t7-t6)
            T_X = row_as_probdist(X_sim, dense_output=False, preserve_all_zero_row=False) # **sparse mat**
            t8 = time.time()
            print('BallTree implementation ALL time',t8-t1)
            del A, X, X_sim, data, row, col, neigh, sim
            

        # ============================================== information fusion via transition matrices =======================================================
        print('about beta, beta_mode, alpha: ', self.beta, self.beta_mode, self.alpha)
        b = None

        # mode 1: fixed beta, except if T_A has any zero rows, set beta=1.0
        if self.beta_mode == 1:
            print('====== fixed beta: T = (1-beta)*T_A + beta*T_X where beta= ', self.beta)
            b = np.array(n * [self.beta])                                # vectored computing
            b[~np.asarray(T_A.sum(axis=1) != 0).ravel()] = 1.0  # if T_A has any zero rows, set beta=0

        # mode 2: adaptive beta baed on average degree which reflects the richness of structural info
        if self.beta_mode == 2:
            print('====== adaptive beta: T = (1-beta)*T_A + beta*T_X, where adaptive beta=(1.0+ave_deg^alpha)^(-1.0/alpha) and alpha= ', self.alpha)
            if self.g.G.is_directed():
                print('directed graph, TODO...')
                exit(0)
            ave_deg = len(self.g.G.edges()) * 2.0 / len(self.g.G.nodes())  # see def http://konect.uni-koblenz.de/statistics/avgdegree
            b = deg2beta_mapping(ave_deg, alpha=self.alpha)                # mapping by the characteristic curve of adaptive beta
            b = np.array(n * [b])                               
            b[~np.asarray(T_A.sum(axis=1) != 0).ravel()] = 1.0

        # mode 3: adaptive beta based on each node degree
        if self.beta_mode == 3:
            print('====== adaptive beta: T = (1-beta)*T_A + beta*T_X, where adaptive beta=(1.0+node_deg^alpha)^(-1.0/alpha) and alpha= ', self.alpha)
            if self.g.G.is_directed():
                print('directed graph, TODO...')
                exit(0)
            node_deg_list = [deg*2 for (node, deg) in self.g.G.degree()]  # *2 due to undirected graph; in consistant with ave_deg after mapping
            b = deg2beta_mapping(node_deg_list, alpha=self.alpha)         # mapping by the characteristic curve of adaptive beta

        T = sparse.diags(1.0-b).dot(T_A) + sparse.diags(b).dot(T_X)
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


# ------------------------ utils draw_characteristic_curve ---------------------------
def draw_characteristic_curve():
    import matplotlib.pyplot as plt
    deg_list = np.arange(0, 100, 0.01)
    beta_list_1 = deg2beta_mapping(deg_list, alpha=0.5)
    beta_list_2 = deg2beta_mapping(deg_list, alpha=1)
    beta_list_3 = deg2beta_mapping(deg_list, alpha=np.e)
    beta_list_4 = deg2beta_mapping(deg_list, alpha=10)
    plt.plot(deg_list, beta_list_1, label='alpha=0.5')
    plt.plot(deg_list, beta_list_2, label='alpha=1')
    plt.plot(deg_list, beta_list_3, label='alpha=np.e')
    plt.plot(deg_list, beta_list_4, label='alpha=10')
    plt.legend()
    plt.show()