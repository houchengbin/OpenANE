# -*- coding: utf-8 -*-
import numpy as np
import time
from numpy import linalg as la
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import Word2Vec
from . import walker
import networkx as nx
from libnrl.utils import *
import multiprocessing

'''
#-----------------------------------------------------------------------------
# author: Chengbin Hou @ SUSTech 2018
# Email: Chengbin.Hou10@foxmail.com
#-----------------------------------------------------------------------------
'''

def multiprocessor_argpartition(vec):
    topk = 20
    print('len of vec...',len(vec))
    return np.argpartition(vec, -topk)[-topk:]


class ABRW(object):

    def __init__(self, graph, dim, alpha, topk, path_length, num_paths, **kwargs):
        self.g = graph
        self.alpha = float(alpha)
        self.topk = int(topk)
        kwargs["workers"] = kwargs.get("workers", 1)

        self.P = self.biasedTransProb() #obtain biased transition probs mat
        weighted_walker = walker.BiasedWalker(g=self.g, P=self.P, workers=kwargs["workers"]) #instance weighted walker
        #generate sentences according to biased transition probs mat P
        sentences = weighted_walker.simulate_walks(num_walks=num_paths, walk_length=path_length)
        
        #skip-gram parameters
        kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = kwargs.get("size", dim)
        kwargs["sg"] = 1  #use skip-gram; but see deepwalk which uses 'hs' = 1
        self.size = kwargs["size"]
        #learning embedding by skip-gram model
        print("Learning representation...")
        word2vec = Word2Vec(**kwargs)
        #save emb for later eval
        self.vectors = {}
        for word in self.g.G.nodes():
            self.vectors[word] = word2vec.wv[word] #save emb
        del word2vec

#----------------------------------------key of our method---------------------------------------------
    def biasedTransProb(self):
        '''
        given: A and X --> P_A and P_X
        research question: how to combine A and X in a more principled way
        genral idea: Attribute Biased Random Walk
        i.e. a walker based on a mixed transition matrix by P=alpha*P_A + (1-alpha)*P_X
        result: ABRW-trainsition matrix; P
        *** questions: 1) what about if we have some single nodes i.e. some rows of P_A gives 0s
                       2) the similarity/distance metric to obtain P_X
                       3) alias sampling as used in node2vec for speeding up, but this is the case 
                            if each row of P gives many 0s 
                            --> how to make each row of P is a pdf and meanwhile is sparse
        '''

        print("obtaining biased transition probs mat...")
        t1 = time.time()

        A = self.g.get_adj_mat()   #adj/struc info mat
        P_A = row_as_probdist(A)  #if single node, return [0, 0, 0 ..] we will fix this later

        X = self.g.get_attr_mat()   #attr info mat
        X_compressed = X    #if need speed up, try to use svd or pca for compression, but will loss some acc
        #X_compressed = self.g.preprocessAttrInfo(X=X, dim=200, method='pca')  #svd or pca for dim reduction; follow TADW setting use svd with dim=200
        from sklearn.metrics.pairwise import linear_kernel, cosine_similarity, cosine_distances, euclidean_distances  # we may try diff metrics
        #ref http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise
        #t1=time.time()
        X_sim = cosine_similarity(X_compressed, X_compressed)
        #t2=time.time()
        #print('======no need pre proce', t2-t1)

        
        #way5: a faster implementation of way5 by Zeyu Dong
        topk = self.topk
        print('way5 remain self---------topk = ', topk)
        t1 = time.time()
        cutoff = np.partition(X_sim, -topk, axis=1)[:,-topk:].min(axis=1)
        X_sim[(X_sim < cutoff)] = 0
        t2 = time.time()


        P_X = row_as_probdist(X_sim)
        t3 = time.time()
        for i in range(P_X.shape[0]):
            sum_row = P_X[i].sum()
            if sum_row != 1.0:          #to avoid some numerical issue...
                delta = 1.0 - sum_row   #delta is very very samll number say 1e-10 or even less...
                P_X[i][i] = P_X[i][i] + delta  #the diagnoal must be largest of the that row + delta --> almost no effect
        t4 = time.time()
        print('topk time: ',t2-t1 ,'row normlize time: ',t3-t2, 'dealing numerical issue time: ', t4-t3)
        del A, X, X_compressed, X_sim
        
        #=====================================core of our idea========================================
        print('------alpha for P = alpha * P_A + (1-alpha) * P_X----: ', self.alpha)
        n = self.g.get_num_nodes()
        P = np.zeros((n,n), dtype=float)
        for i in range(n):
            if (P_A[i] == 0).all():  #single node case if the whole row are 0s
            #if P_A[i].sum() == 0:
                P[i] = P_X[i]        #use 100% attr info to compensate 
            else:                    #non-single node case; use (1.0-self.alpha) attr info to compensate
                P[i] = self.alpha * P_A[i] + (1.0-self.alpha) * P_X[i]
        print('# of single nodes for P_A: ', n - P_A.sum(axis=1).sum(), ' # of non-zero entries of P_A: ', np.count_nonzero(P_A))
        print('# of single nodes for P_X: ', n - P_X.sum(axis=1).sum(), ' # of non-zero entries of P_X: ', np.count_nonzero(P_X))
        t5 = time.time()
        print('ABRW biased transition prob preprocessing time: {:.2f}s'.format(t5-t4))
        return P


    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()