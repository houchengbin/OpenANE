# -*- coding: utf-8 -*-
import time

import numpy as np
from scipy import sparse

# from sklearn.model_selection import train_test_split


'''
#-----------------------------------------------------------------------------
# Chengbin Hou @ SUSTech 2018
# Email: Chengbin.Hou10@foxmail.com
#-----------------------------------------------------------------------------
'''

# ---------------------------------ulits for calculation--------------------------------


def row_as_probdist(mat, dense_output=False):
    """Make each row of matrix sums up to 1.0, i.e., a probability distribution.
    Support both dense and sparse matrix.

    Attributes
    ----------
    mat : scipy sparse matrix or dense matrix or numpy array
        The matrix to be normalized
    dense_output : bool
        whether forced dense output
    Note
    ----
    For row with all entries 0, we normalize it to a vector with all entries 1/n

    Returns
    -------
    dense or sparse matrix:
        return dense matrix if input is dense matrix or numpy array
        return sparse matrix for sparse matrix input
        (note: np.array & np.matrix are diff; and may cause some dim issues...)
    """
    row_sum = np.array(mat.sum(axis=1)).ravel()  # type: np.array
    zero_rows = row_sum == 0
    row_sum[zero_rows] = 1
    diag = sparse.dia_matrix((1 / row_sum, 0), (mat.shape[0], mat.shape[0]))
    mat = diag.dot(mat)
    mat += sparse.csr_matrix(zero_rows.astype(int)).T.dot(sparse.csr_matrix(np.repeat(1 / mat.shape[1], mat.shape[1])))

    if dense_output and sparse.issparse(mat):
        return mat.todense()
    return mat


def pairwise_similarity(mat, type='cosine'):
    # XXX: possible to integrate pairwise_similarity with top_k to enhance performance? 
    # we'll use it elsewhere. if really needed, write a new method for this purpose
    if type == 'cosine':  # support sprase and dense mat
        from sklearn.metrics.pairwise import cosine_similarity
        result = cosine_similarity(mat, dense_output=True)
    elif type == 'jaccard':
        from sklearn.metrics import jaccard_similarity_score
        from sklearn.metrics.pairwise import pairwise_distances
        # n_jobs=-1 means using all CPU for parallel computing
        result = pairwise_distances(mat.todense(), metric=jaccard_similarity_score, n_jobs=-1)
    elif type == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        # note: similarity = - distance
        # other version: similarity = 1 - 2 / pi * arctan(distance)
        result = euclidean_distances(mat)
        result = -result
        # result = 1 - 2 / np.pi * np.arctan(result)
    elif type == 'manhattan':
        from sklearn.metrics.pairwise import manhattan_distances
        # note: similarity = - distance
        # other version: similarity = 1 - 2 / pi * arctan(distance)
        result = manhattan_distances(mat)
        result = -result
        # result = 1 - 2 / np.pi * np.arctan(result)
    else:
        print('Please choose from: cosine, jaccard, euclidean or manhattan')
        return 'Not found!'
    return result


# ---------------------------------ulits for preprocessing--------------------------------
def node_auxi_to_attr(fin, fout):
    """ TODO...
        -> read auxi info associated with each node;
        -> preprocessing auxi via:
            1) NLP for sentences; or 2) one-hot for discrete features;
        -> then becomes node attr with m dim, and store them into attr file
    """
    # https://radimrehurek.com/gensim/apiref.html
    # word2vec, doc2vec, 把句子转为vec
    # text2vec, tfidf, 把离散的features转为vec
    pass


def simulate_incomplete_stru():
    pass


def simulate_incomplete_attr():
    pass


def simulate_noisy_world():
    pass

# ---------------------------------ulits for downstream tasks--------------------------------
# XXX: read and save using panda or numpy


def read_edge_label_downstream(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        line = fin.readline()
        if line == '':
            break
        vec = line.strip().split(' ')
        X.append(vec[:2])
        Y.append(vec[2])
    fin.close()
    return X, Y


def read_node_label_downstream(filename):
    """ may be used in node classification task;
        part of labels for training clf and
        the result served as ground truth;
        note: similar method can be found in graph.py -> read_node_label
    """
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        line = fin.readline()
        if line == '':
            break
        vec = line.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


def store_embedddings(vectors, filename, dim):
    """ store embeddings to file
    """
    fout = open(filename, 'w')
    num_nodes = len(vectors.keys())
    fout.write("{} {}\n".format(num_nodes, dim))
    for node, vec in vectors.items():
        fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
    fout.close()
    print('store the resulting embeddings in file: ', filename)


def load_embeddings(filename):
    """ load embeddings from file
    """
    fin = open(filename, 'r')
    num_nodes, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        line = fin.readline()
        if line == '':
            break
        vec = line.strip().split(' ')
        assert len(vec) == size + 1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == num_nodes
    return vectors


#----------------- 以下你整理到utils，有问题的我都用中文写出来了，没有中文的暂时没啥问题，可以先不用管-----------------------
def generate_edges_for_linkpred(graph, edges_removed, balance_ratio=1.0):
    ''' given a graph and edges_removed;
        generate non_edges not in [both graph and edges_removed];
        return all_test_samples including [edges_removed (pos samples), non_edges (neg samples)];
        return format X=[[1,2],[2,4],...] Y=[1,0,...] where Y tells where corresponding element has a edge
    '''
    g = graph
    num_edges_removed = len(edges_removed)
    num_non_edges = int(balance_ratio * num_edges_removed)
    num = 0
    #np.random.seed(2018)
    non_edges = []
    exist_edges = list(g.G.edges())+list(edges_removed)
    while num < num_non_edges:
        non_edge = list(np.random.choice(g.look_back_list, size=2, replace=False))
        if non_edge not in exist_edges:
            num += 1
            non_edges.append(non_edge)
    
    test_node_pairs = edges_removed + non_edges
    test_edge_labels = list(np.ones(num_edges_removed)) + list(np.zeros(num_non_edges))
    return test_node_pairs, test_edge_labels


def dim_reduction(mat, dim=128, method='pca'):
    import time
    ''' dimensionality reduction: PCA, SVD, etc...
        dim = # of columns
    '''
    print('START dimensionality reduction using ' + method + ' ......')
    t1 = time.time()
    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=dim, svd_solver='auto', random_state=None)
        mat_reduced = pca.fit_transform(mat)   #sklearn pca auto remove mean, no need to preprocess
    elif method == 'svd':
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=dim, n_iter=5, random_state=None)
        mat_reduced = svd.fit_transform(mat)
    else:  #to do... more methods... e.g. random projection, ica, t-sne...
        print('dimensionality reduction method not found......')
    t2 = time.time()
    print('END dimensionality reduction: {:.2f}s'.format(t2-t1))
    return mat_reduced


def row_normalized(mat, is_transition_matrix=False):
    ''' to do...
        两个问题：1）sparse矩阵在该场景下比dense慢,(至少我自己写的这块代码是)
                2）dense矩阵测试后发现所有元素加起来不是整数，似乎还是要用我以前笨方法来弥补
                3)在is_transition_matrix时候，需要给全零行赋值，sparse时候会有点小问题，不能直接mat[i, :] = p赋值
    '''
    p = 1.0/mat.shape[0] #probability = 1/num of rows
    norms = np.asarray(mat.sum(axis=1)).ravel()
    for i, norm in enumerate(norms):
        if norm != 0:
            mat[i, :] /= norm
        else:
            if is_transition_matrix:
                mat[i, :] = p #every row of transition matrix should sum up to 1
            else:
                pass #do nothing; keep all-zero row
    return mat

''' 笨方法如下'''
def rowAsPDF(mat): #make each row sum up to 1 i.e. a probabolity density distribution
    mat = np.array(mat)
    for i in range(mat.shape[0]):
        sum_row = mat[i,:].sum()
        if sum_row !=0:
            mat[i,:] = mat[i,:]/sum_row     #if a row [0, 1, 1, 1] -> [0, 1/3, 1/3, 1/3] -> may have some small issue...
        else:
            # to do...
            # for node without any link... remain row as [0, 0, 0, 0]  OR set to [1/n, 1/n, 1/n...]??
            pass 
        if mat[i,:].sum() != 1.00:      #small trick to make sure each row is a pdf 笨犯法。。。
            error = 1.00 - mat[i,:].sum()
            mat[i,-1] += error
    return mat



def sparse_to_dense():
    ''' to dense np.matrix format 你补充下，记得dtype用float64'''
    import scipy.sparse as sp
    pass

def dense_to_sparse():
    ''' to sparse crs format 你补充下，记得dtype用float64'''
    import scipy.sparse as sp
    pass
