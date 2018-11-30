'''
ANE method: Attributed Social Network Embedding (ASNE)

modified by Chengbin Hou 2018
1) convert OpenANE data format to ASNE data format
2) compatible with latest tensorflow 1.10.0
3) add early stopping
4) as ASNE paper stated, we add two hidden layers with softsign activation func

part of code was originally forked from https://github.com/lizi-git/ASNE
'''

import math
import time

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin


class ASNE(BaseEstimator, TransformerMixin):
    def __init__(self, graph, dim, alpha=1.0, learning_rate=0.0001, batch_size=128, epoch=20, n_neg_samples=10,
                 early_stopping=2000):  # it seems that overfitting can get better result? try other early_stopping... to do...

        t1 = time.time()
        X, nodes, id_N, attr_M, id_embedding_size, attr_embedding_size = format_data_from_OpenANE_to_ASNE(g=graph, dim=dim)
        t2 = time.time()
        print(f'transform data format from OpenANE to ASNE; time cost: {(t2-t1):.2f}s')

        self.node_N = id_N  # n
        self.attr_M = attr_M  # m
        self.X_train = X  # {'data_id_list': [], 'data_label_list': [], 'data_attr_list': []}
        self.nodes = nodes  # {'node_id': [], 'node_attr: []'}
        self.id_embedding_size = id_embedding_size      # set to dim/2
        self.attr_embedding_size = attr_embedding_size  # set to dim/2
        self.vectors = {}  # final embs
        self.look_back_list = graph.look_back_list  # from OpenANE data stcuture

        self.alpha = alpha  # set to 1.0 by default
        self.n_neg_samples = n_neg_samples  # set to 10 by default
        self.batch_size = batch_size  # set to 128 by default
        self.learning_rate = learning_rate
        self.epoch = epoch  # set to 20 by default

        self._init_graph()  # init all variables in a tensorflow graph
        self.early_stopping = early_stopping  # early stopping if training loss increased for xx iterations
        self.train()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        # with self.graph.as_default(), tf.device('/gpu:0'):
        with self.graph.as_default():
            # Set graph level random seed
            # tf.set_random_seed(2018)
            # Input data.
            self.train_data_id = tf.placeholder(tf.int32, shape=[None])                   # batch_size * 1
            self.train_data_attr = tf.placeholder(tf.float32, shape=[None, self.attr_M])  # batch_size * attr_M
            self.train_labels = tf.placeholder(tf.int32, shape=[None, 1])                 # batch_size * 1

            # Variables.
            network_weights = self._initialize_weights()
            self.weights = network_weights

            # Model.
            # Look up embeddings for node_id.
            self.id_embed = tf.nn.embedding_lookup(self.weights['in_embeddings'], self.train_data_id)  # batch_size * id_dim
            self.attr_embed = tf.matmul(self.train_data_attr, self.weights['attr_embeddings'])        # batch_size * attr_dim
            self.embed_layer = tf.concat([self.id_embed, self.alpha * self.attr_embed], 1)             # batch_size * (id_dim + attr_dim) #an error due to old tf!

            '''
            ## can add hidden_layers component here!----------------------------------
            #0) no hidden layer
            #1) 128
            #2) 256+128  ##--------paper stated it used two hidden layers with softsign
            #3) 512+256+128
            len_h1_in = self.id_embedding_size + self.attr_embedding_size
            len_h1_out = 256 #or self.id_embedding_size + self.attr_embedding_size # if only add h1
            len_h2_in = len_h1_out
            len_h2_out = self.id_embedding_size + self.attr_embedding_size
            self.h1 = add_layer(inputs=self.embed_layer, in_size=len_h1_in, out_size=len_h1_out, activation_function=tf.nn.softsign)
            self.h2 = add_layer(inputs=self.h1, in_size=len_h2_in, out_size=len_h2_out, activation_function=tf.nn.softsign)
            ## -------------------------------------------------------------------------
            '''

            # Compute the loss, using a sample of the negative labels each time.
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=self.weights['out_embeddings'], biases=self.weights['biases'],  # if one needs to change layers
                                                                  inputs=self.embed_layer, labels=self.train_labels, num_sampled=self.n_neg_samples, num_classes=self.node_N))  # try inputs = self.embed_layer or self.h1 or self.h2 or ...
            # Optimizer.
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            # print("AdamOptimizer")

            # init
            init = tf.initialize_all_variables()
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['in_embeddings'] = tf.Variable(tf.random_uniform([self.node_N, self.id_embedding_size], -1.0, 1.0))    # id_N * id_dim
        all_weights['attr_embeddings'] = tf.Variable(tf.random_uniform([self.attr_M, self.attr_embedding_size], -1.0, 1.0))  # attr_M * attr_dim
        all_weights['out_embeddings'] = tf.Variable(tf.truncated_normal([self.node_N, self.id_embedding_size + self.attr_embedding_size],
                                                                        stddev=1.0 / math.sqrt(self.id_embedding_size + self.attr_embedding_size)))
        all_weights['biases'] = tf.Variable(tf.zeros([self.node_N]))
        return all_weights

    def partial_fit(self, X):  # fit a batch
        feed_dict = {self.train_data_id: X['batch_data_id'], self.train_data_attr: X['batch_data_attr'],
                     self.train_labels: X['batch_data_label']}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):  # useless for a moment...
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]

    def train(self):
        self.Embeddings = []
        total_batch = int(len(self.X_train['data_id_list']) / self.batch_size)
        iter_count = 0
        train_loss_best = 0
        train_loss_keep_increasing = 0
        early_stopping = self.early_stopping  # early stopping if training loss increased

        for epoch in range(self.epoch):
            t1 = time.time()

            for i in range(total_batch):
                # generate a batch data
                batch_xs = {}
                start_index = np.random.randint(0, len(self.X_train['data_id_list']) - self.batch_size)
                batch_xs['batch_data_id'] = self.X_train['data_id_list'][start_index:(start_index + self.batch_size)]  # generate batch data
                batch_xs['batch_data_attr'] = self.X_train['data_attr_list'][start_index:(start_index + self.batch_size)]
                batch_xs['batch_data_label'] = self.X_train['data_label_list'][start_index:(start_index + self.batch_size)]

                # Fit training using batch data
                train_loss = self.partial_fit(batch_xs)
                iter_count += 1
                if iter_count == 1:
                    train_loss_best = train_loss
                else:
                    if train_loss_best > train_loss:   # training loss decreasing
                        train_loss_best = train_loss
                        train_loss_keep_increasing = 0  # reset
                    else:                              # training loss increasing
                        train_loss_keep_increasing += 1
                        if train_loss_keep_increasing > early_stopping:  # early stopping
                            print(f'early stopping @ iter {iter_count}; take out embs and return')
                            Embeddings_out = self.getEmbedding('out_embedding', self.nodes)
                            Embeddings_in = self.getEmbedding('embed_layer', self.nodes)
                            self.Embeddings = Embeddings_out + Embeddings_in  # simply mean them and as final embedding; try concat? to do...
                            ind = 0
                            for id in self.nodes['node_id']:  # self.nodes['node_id']=self.look_back_list
                                self.vectors[id] = self.Embeddings[ind]
                                ind += 1
                            return self.vectors
                        else:
                            pass
            t2 = time.time()
            print(f'epoch @ {epoch+1}/{self.epoch}; time cost: {(t2-t1):.2f}s',)

        print(f'finish all {self.epoch} epochs; take out embs and return')
        Embeddings_out = self.getEmbedding('out_embedding', self.nodes)
        Embeddings_in = self.getEmbedding('embed_layer', self.nodes)
        self.Embeddings = Embeddings_out + Embeddings_in  # simply mean them and as final embedding; try concat? to do...
        ind = 0
        for id in self.nodes['node_id']:  # self.nodes['node_id']=self.look_back_list
            self.vectors[id] = self.Embeddings[ind]
            ind += 1
        return self.vectors

    def getEmbedding(self, type, nodes):
        if type == 'embed_layer':
            feed_dict = {self.train_data_id: nodes['node_id'], self.train_data_attr: nodes['node_attr']}
            Embedding = self.sess.run(self.embed_layer, feed_dict=feed_dict)
            return Embedding
        if type == 'out_embedding':
            Embedding = self.sess.run(self.weights['out_embeddings'])  # sess.run to get embeddings from tf
            return Embedding  # nodes_number * (id_dim + attr_dim)

    def save_embeddings(self, filename):
        '''
        save embeddings to file
        '''
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.id_embedding_size+self.attr_embedding_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()

# ---------------------------------------------- ASNE utils ------------------------------------------------


def format_data_from_OpenANE_to_ASNE(g, dim):
    ''' convert OpenANE data format to ASNE data format '''
    attr_Matrix = g.get_attr_mat(is_sparse=False)
    id_N = attr_Matrix.shape[0]  # n nodes
    attr_M = attr_Matrix.shape[1]  # m features

    X = {}
    X['data_id_list'] = []
    X['data_label_list'] = []
    X['data_attr_list'] = []
    edgelist = [edge for edge in g.G.edges]
    print('If an edge only have one direction, double it......')
    cnt = 0
    for edge in edgelist:  # traning sample = start node, end node, start node attr
        X['data_id_list'].append(edge[0])
        X['data_label_list'].append(edge[1])
        X['data_attr_list'].append(attr_Matrix[g.look_up_dict[edge[0]]][:])
        cnt += 1
        if (edge[1], edge[0]) not in edgelist:  # double! as paper said--------------
            X['data_id_list'].append(edge[1])
            X['data_label_list'].append(edge[0])
            X['data_attr_list'].append(attr_Matrix[g.look_up_dict[edge[1]]][:])
            cnt += 1
    print(f'edges before doubling: {g.get_num_edges()}')
    print(f'edges after doubling: {cnt}')

    X['data_id_list'] = np.array(X['data_id_list']).reshape(-1).astype(int)
    X['data_label_list'] = np.array(X['data_label_list']).reshape(-1, 1).astype(int)
    X['data_attr_list'] = np.array(X['data_attr_list']).reshape(cnt, attr_M)

    nodes = {}
    nodes['node_id'] = g.look_back_list
    nodes['node_attr'] = attr_Matrix

    id_embedding_size = int(dim/2)
    attr_embedding_size = int(dim/2)
    print('id_embedding_size', id_embedding_size, '\nattr_embedding_size', attr_embedding_size)
    return X, nodes, id_N, attr_M, id_embedding_size, attr_embedding_size


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_uniform([in_size, out_size], -1.0, 1.0))  # init as paper stated
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
