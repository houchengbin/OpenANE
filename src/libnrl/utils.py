"""
commonly used utils
by Chengbin Hou & Zeyu Dong
"""

import time

import numpy as np
from scipy import sparse


# ---------------------------------ulits for calculation--------------------------------

def row_as_probdist(mat, dense_output=True, preserve_all_zero_row=False):
    """Make each row of matrix sums up to 1.0, i.e., a probability distribution.
    Support both dense and sparse matrix.

    Attributes
    ----------
    mat : scipy sparse matrix or dense matrix or numpy array
        The matrix to be normalized
    dense_output : bool
        whether forced dense output
    perserve_zeros : bool
        If False, for row with all entries 0, we normalize it to a vector with all entries 1/n.
        Leave 0 otherwise
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
    if not preserve_all_zero_row:
        print('For all-zero row, replace each 0 with value 1/dim(row)... not preserving zero i.e. a strict transition matrix')
        mat += sparse.csr_matrix(zero_rows.astype(int)).T.dot(sparse.csr_matrix(np.repeat(1 / mat.shape[1], mat.shape[1])))

    if dense_output and sparse.issparse(mat):
        return mat.todense()
    return mat


def pairwise_similarity(mat, type='cosine'):   # for efficiency, plz given dense mat as the input
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
        result = euclidean_distances(mat)
        result = -result
    elif type == 'manhattan':
        from sklearn.metrics.pairwise import manhattan_distances
        # note: similarity = - distance
        result = manhattan_distances(mat)
        result = -result
    else:
        print('Please choose from: cosine, jaccard, euclidean or manhattan')
        return 'Not found!'
    return result


# ---------------------------------ulits for downstream tasks--------------------------------

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


# ---------------------------------others--------------------------------

def dim_reduction(mat, dim=128, method='pca'):
    ''' dimensionality reduction: PCA, SVD, etc...
        dim = # of columns
    '''
    print('START dimensionality reduction using ' + method + ' ......')
    t1 = time.time()
    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=dim, svd_solver='auto', random_state=None)
        mat_reduced = pca.fit_transform(mat)  # sklearn pca auto remove mean, no need to preprocess
    elif method == 'svd':
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=dim, n_iter=5, random_state=None)
        mat_reduced = svd.fit_transform(mat)
    else:  # to do... more methods... e.g. random projection, ica, t-sne...
        print('dimensionality reduction method not found......')
    t2 = time.time()
    print('END dimensionality reduction: {:.2f}s'.format(t2-t1))
    return mat_reduced
