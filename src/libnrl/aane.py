"""
ANE method: Accelerated Attributed Network Embedding (AANE)

modified by Chengbin Hou 2018

originally from https://github.com/xhuang31/AANE_Python
"""

import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from math import ceil

class AANE:
    """Jointly embed Net and Attri into embedding representation H
    H = AANE(Net,Attri,d).function()
    H = AANE(Net,Attri,d,lambd,rho).function()
    H = AANE(Net,Attri,d,lambd,rho,maxiter).function()
    H = AANE(Net,Attri,d,lambd,rho,maxiter,'Att').function()
    H = AANE(Net,Attri,d,lambd,rho,maxiter,'Att',splitnum).function()
    :param Net: the weighted adjacency matrix
    :param Attri: the attribute information matrix with row denotes nodes
    :param d: the dimension of the embedding representation
    :param lambd: the regularization parameter
    :param rho: the penalty parameter
    :param maxiter: the maximum number of iteration
    :param 'Att': refers to conduct Initialization from the SVD of Attri
    :param splitnum: the number of pieces we split the SA for limited cache
    :return: the embedding representation H
    Copyright 2017 & 2018, Xiao Huang and Jundong Li.
    $Revision: 1.0.2 $  $Date: 2018/02/19 00:00:00 $
    """
    def __init__(self, graph, dim, lambd=0.05, rho=5, maxiter=5, mode='comb', *varargs):
        self.dim = dim
        self.look_back_list = graph.look_back_list #look back node id for Net and Attr
        self.lambd = lambd  # Initial regularization parameter
        self.rho = rho  # Initial penalty parameter
        self.maxiter = maxiter  # Max num of iteration
        splitnum = 1  # number of pieces we split the SA for limited cache
        if mode == 'comb':
            print('==============AANE-comb mode: jointly learn emb from both structure and attribute info========')
            Net = graph.get_adj_mat()
            Attri = graph.get_attr_mat()
        elif mode == 'pure':
            print('======================AANE-pure mode: learn emb purely from structure info====================')
            Net = graph.get_adj_mat()
            Attri = Net
        else:
            exit(0)
        
        [self.n, m] = Attri.shape  # n = Total num of nodes, m = attribute category num
        Net = sparse.lil_matrix(Net)
        Net.setdiag(np.zeros(self.n))
        Net = csc_matrix(Net)
        Attri = csc_matrix(Attri)
        if len(varargs) >= 4 and varargs[3] == 'Att':
            sumcol = np.arange(m)
            np.random.shuffle(sumcol)
            self.H = svds(Attri[:, sumcol[0:min(10 * self.dim, m)]], self.dim)[0]
        else:
            sumcol = Net.sum(0)
            self.H = svds(Net[:, sorted(range(self.n), key=lambda k: sumcol[0, k], reverse=True)[0:min(10 * self.dim, self.n)]], self.dim)[0]

        if len(varargs) > 0:
            self.lambd = varargs[0]
            self.rho = varargs[1]
            if len(varargs) >= 3:
                self.maxiter = varargs[2]
                if len(varargs) >= 5:
                    splitnum = varargs[4]
        self.block = min(int(ceil(float(self.n) / splitnum)), 7575)  # Treat at least each 7575 nodes as a block
        self.splitnum = int(ceil(float(self.n) / self.block))
        with np.errstate(divide='ignore'):  # inf will be ignored
            self.Attri = Attri.transpose() * sparse.diags(np.ravel(np.power(Attri.power(2).sum(1), -0.5)))
        self.Z = self.H.copy()
        self.affi = -1  # Index for affinity matrix sa
        self.U = np.zeros((self.n, self.dim))
        self.nexidx = np.split(Net.indices, Net.indptr[1:-1])
        self.Net = np.split(Net.data, Net.indptr[1:-1])

        self.vectors = {}
        self.function()  #run aane----------------------------


    '''################# Update functions #################'''
    def updateH(self):
        xtx = np.dot(self.Z.transpose(), self.Z) * 2 + self.rho * np.eye(self.dim)
        for blocki in range(self.splitnum):  # Split nodes into different Blocks
            indexblock = self.block * blocki  # Index for splitting blocks
            if self.affi != blocki:
                self.sa = self.Attri[:, range(indexblock, indexblock + min(self.n - indexblock, self.block))].transpose() * self.Attri
                self.affi = blocki
            sums = self.sa.dot(self.Z) * 2
            for i in range(indexblock, indexblock + min(self.n - indexblock, self.block)):
                neighbor = self.Z[self.nexidx[i], :]  # the set of adjacent nodes of node i
                for j in range(1):
                    normi_j = np.linalg.norm(neighbor - self.H[i, :], axis=1)  # norm of h_i^k-z_j^k
                    nzidx = normi_j != 0  # Non-equal Index
                    if np.any(nzidx):
                        normi_j = (self.lambd * self.Net[i][nzidx]) / normi_j[nzidx]
                        self.H[i, :] = np.linalg.solve(xtx + normi_j.sum() * np.eye(self.dim), sums[i - indexblock, :] + (
                                    neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0) + self.rho * (
                                                                   self.Z[i, :] - self.U[i, :]))
                    else:
                        self.H[i, :] = np.linalg.solve(xtx, sums[i - indexblock, :] + self.rho * (
                                    self.Z[i, :] - self.U[i, :]))
    def updateZ(self):
        xtx = np.dot(self.H.transpose(), self.H) * 2 + self.rho * np.eye(self.dim)
        for blocki in range(self.splitnum):  # Split nodes into different Blocks
            indexblock = self.block * blocki  # Index for splitting blocks
            if self.affi != blocki:
                self.sa = self.Attri[:, range(indexblock, indexblock + min(self.n - indexblock, self.block))].transpose() * self.Attri
                self.affi = blocki
            sums = self.sa.dot(self.H) * 2
            for i in range(indexblock, indexblock + min(self.n - indexblock, self.block)):
                neighbor = self.H[self.nexidx[i], :]  # the set of adjacent nodes of node i
                for j in range(1):
                    normi_j = np.linalg.norm(neighbor - self.Z[i, :], axis=1)  # norm of h_i^k-z_j^k
                    nzidx = normi_j != 0  # Non-equal Index
                    if np.any(nzidx):
                        normi_j = (self.lambd * self.Net[i][nzidx]) / normi_j[nzidx]
                        self.Z[i, :] = np.linalg.solve(xtx + normi_j.sum() * np.eye(self.dim), sums[i - indexblock, :] + (
                                    neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0) + self.rho * (
                                                                   self.H[i, :] + self.U[i, :]))
                    else:
                        self.Z[i, :] = np.linalg.solve(xtx, sums[i - indexblock, :] + self.rho * (
                                    self.H[i, :] + self.U[i, :]))

    def function(self):
        self.updateH()
        '''################# Iterations #################'''
        for i in range(self.maxiter):
            import time
            t1=time.time()
            self.updateZ()
            self.U = self.U + self.H - self.Z
            self.updateH()
            t2=time.time()
            print(f'iter: {i+1}/{self.maxiter}; time cost {t2-t1:0.2f}s')

        #-------save emb to self.vectors and return
        ind = 0
        for id in self.look_back_list:
            self.vectors[id] = self.H[ind]
            ind += 1
        return self.vectors
    
    def save_embeddings(self, filename):
        '''
        save embeddings to file
        '''
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.dim))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,' '.join([str(x) for x in vec])))
        fout.close()
