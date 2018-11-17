# -*- coding: utf-8 -*-
'''
Tensorflow implementation of Social Network Embedding framework (SNE)
@author: Lizi Liao (liaolizi.llz@gmail.com)
part of code was originally forked from https://github.com/lizi-git/ASNE

modified by Chengbin Hou 2018
1) convert OpenANE data format to ASNE data format
2) compatible with latest tensorflow 1.2
3) add more comments
4) support eval testing set during each xx epoches
5) as ASNE paper stated, we add two hidden layers with softsign activation func
'''

import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from .classify import ncClassifier, lpClassifier, read_node_label
from sklearn.linear_model import LogisticRegression

def format_data_from_OpenANE_to_ASNE(g, dim):
    '''
    convert OpenANE data format to ASNE data format
    g: OpenANE graph data structure
    dim: final embedding dim
    '''
    attr_Matrix = g.getX()
    #attr_Matrix = g.preprocessAttrInfo(attr_Matrix, dim=200, method='svd') #similar to aane, the same preprocessing
    #print('with this preprocessing, ASNE can get better result, as well as, faster speed----------------')
    id_N = attr_Matrix.shape[0]    #n nodes
    attr_M = attr_Matrix.shape[1]  #m features

    edge_num = len(g.G.edges)                           #total edges for traning
    X={}                                                #one-to-one correspondence
    X['data_id_list'] = np.zeros(edge_num)              #start node list for traning
    X['data_label_list'] = np.zeros(edge_num)           #end node list for training
    X['data_attr_list'] = np.zeros([edge_num, attr_M])  #attr corresponds to start node
    edgelist = [edge for edge in g.G.edges]
    i = 0
    for edge in edgelist:      #traning sample = start node, end node, start node attr
        X['data_id_list'][i] = edge[0]
        X['data_label_list'][i] = edge[1]
        X['data_attr_list'][i] = attr_Matrix[ g.look_up_dict[edge[0]] ][:]
        i += 1
    X['data_id_list'] = X['data_id_list'].reshape(-1).astype(int)
    X['data_label_list'] = X['data_label_list'].reshape(-1,1).astype(int)

    nodes={}                                 #one-to-one correspondence
    nodes['node_id'] = g.look_back_list      #n nodes
    nodes['node_attr'] = list(attr_Matrix)   #m features -> n*m

    id_embedding_size = int(dim/2)
    attr_embedding_size = int(dim/2)
    print('id_embedding_size', id_embedding_size, 'attr_embedding_size', attr_embedding_size)
    return X, nodes, id_N, attr_M, id_embedding_size, attr_embedding_size


def add_layer(inputs, in_size, out_size, activation_function=None):
   # add one more layer and return the output of this layer
   Weights = tf.Variable(tf.random_uniform([in_size, out_size], -1.0, 1.0)) #init as paper stated
   biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
   Wx_plus_b = tf.matmul(inputs, Weights) + biases
   if activation_function is None:
       outputs = Wx_plus_b
   else:
       outputs = activation_function(Wx_plus_b)
   return outputs


class ASNE(BaseEstimator, TransformerMixin):
    def __init__(self, graph, dim, alpha = 1.0, batch_size=128, learning_rate=0.001, 
                  n_neg_samples=10, epoch=100, random_seed=2018, X_test=0, Y_test=0, task='nc', nc_ratio=0.5, lp_ratio=0.9, label_file=''):
        # bind params to class
        X, nodes, id_N, attr_M, id_embedding_size, attr_embedding_size = format_data_from_OpenANE_to_ASNE(g=graph, dim=dim)
        self.node_N = id_N      #n
        self.attr_M = attr_M    #m
        self.X_train = X        #{'data_id_list': [], 'data_label_list': [], 'data_attr_list': []}
        self.nodes = nodes      #{'node_id': [], 'node_attr: []'}
        self.id_embedding_size = id_embedding_size      # set to dim/2
        self.attr_embedding_size = attr_embedding_size  # set to dim/2
        self.vectors = {}
        self.dim = dim
        self.look_back_list = graph.look_back_list  #from OpenANE data stcuture
        
        self.alpha = alpha                   #set to 1.0 by default
        self.n_neg_samples = n_neg_samples   #set to 10 by default
        self.batch_size = batch_size         #set to 128 by default
        self.learning_rate = learning_rate
        self.epoch = epoch                   #set to 20 by default
        self.random_seed = random_seed   
        self._init_graph()   #init all variables in a tensorflow graph

        self.task = task
        self.nc_ratio = nc_ratio
        self.lp_ratio = lp_ratio
        if self.task == 'lp':         #if not lp task, we do not need to keep testing edges
            self.X_test = X_test
            self.Y_test = Y_test
            self.train()         #train our tf asne model-----------------
        elif self.task == 'nc' or self.task == 'nclp':
            self.X_nc_label, self.Y_nc_label = read_node_label(label_file)
            self.train()         #train our tf asne model-----------------

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        #with self.graph.as_default(), tf.device('/gpu:0'):
        with self.graph.as_default():
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.train_data_id = tf.placeholder(tf.int32, shape=[None])                   # batch_size * 1
            self.train_data_attr = tf.placeholder(tf.float32, shape=[None, self.attr_M])  # batch_size * attr_M
            self.train_labels = tf.placeholder(tf.int32, shape=[None, 1])                 # batch_size * 1

            # Variables.
            network_weights = self._initialize_weights()
            self.weights = network_weights
            
            # Model.
            # Look up embeddings for node_id.
            self.id_embed =  tf.nn.embedding_lookup(self.weights['in_embeddings'], self.train_data_id) # batch_size * id_dim
            self.attr_embed =  tf.matmul(self.train_data_attr, self.weights['attr_embeddings'])        # batch_size * attr_dim
            self.embed_layer = tf.concat([self.id_embed, self.alpha * self.attr_embed], 1)             # batch_size * (id_dim + attr_dim) #an error due to old tf!

            
            ## can add hidden_layers component here!
            #0) no hidden layer
            #1) 128
            #2) 256+128  ##--------paper stated it used two hidden layers with activation function softsign....
            #3) 512+256+128
            len_h1_in = self.id_embedding_size+self.attr_embedding_size
            len_h1_out = 256
            len_h2_in = len_h1_out
            len_h2_out = 128
            self.h1 = add_layer(inputs=self.embed_layer, in_size=len_h1_in, out_size=len_h1_out, activation_function=tf.nn.softsign)
            self.h2 = add_layer(inputs=self.h1, in_size=len_h2_in, out_size=len_h2_out, activation_function=tf.nn.softsign)
            

            # Compute the loss, using a sample of the negative labels each time.
            self.loss =  tf.reduce_mean(tf.nn.sampled_softmax_loss(weights = self.weights['out_embeddings'], biases = self.weights['biases'], 
                              inputs = self.h2, labels = self.train_labels, num_sampled = self.n_neg_samples, num_classes=self.node_N))
            # Optimizer.
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)  #tune these parameters?
            # print("AdamOptimizer")

            # init
            init = tf.initialize_all_variables()
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['in_embeddings'] = tf.Variable(tf.random_uniform([self.node_N, self.id_embedding_size], -1.0, 1.0))    # id_N * id_dim
        all_weights['attr_embeddings'] = tf.Variable(tf.random_uniform([self.attr_M,self.attr_embedding_size], -1.0, 1.0)) # attr_M * attr_dim
        all_weights['out_embeddings'] = tf.Variable(tf.truncated_normal([self.node_N, self.id_embedding_size + self.attr_embedding_size],
                                    stddev=1.0 / math.sqrt(self.id_embedding_size + self.attr_embedding_size)))
        all_weights['biases'] = tf.Variable(tf.zeros([self.node_N]))
        return all_weights

    def partial_fit(self, X): # fit a batch
        feed_dict = {self.train_data_id: X['batch_data_id'], self.train_data_attr: X['batch_data_attr'],
                     self.train_labels: X['batch_data_label']}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):  #useless for a moment...
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]

    def train(self): # fit a dataset
        self.Embeddings = []
        print('Using in + out embedding')

        for epoch in range( self.epoch ):
            total_batch = int( len(self.X_train['data_id_list']) / self.batch_size) #total_batch*batch_size = numOFlinks??
            # print('total_batch in 1 epoch: ', total_batch)
            # Loop over all batches
            for i in range(total_batch):
                # generate a batch data
                batch_xs = {}
                start_index = np.random.randint(0, len(self.X_train['data_id_list']) - self.batch_size)  
                batch_xs['batch_data_id'] = self.X_train['data_id_list'][start_index:(start_index + self.batch_size)] #generate batch data
                batch_xs['batch_data_attr'] = self.X_train['data_attr_list'][start_index:(start_index + self.batch_size)]
                batch_xs['batch_data_label'] = self.X_train['data_label_list'][start_index:(start_index + self.batch_size)]

                # Fit training using batch data
                cost = self.partial_fit(batch_xs)
            
            # Display logs per epoch
            Embeddings_out = self.getEmbedding('out_embedding', self.nodes)
            Embeddings_in = self.getEmbedding('embed_layer', self.nodes)
            self.Embeddings = Embeddings_out + Embeddings_in  #simply mean them and as final embedding; try concat? to do...
            #print('training tensorflow asne model, epoc: ', epoch+1 , ' / ', self.epoch)
            #to save training time, we delete eval testing data @ each epoch
        
            #-----------for each xx epoches; save embeddings {node_id1: [], node_id2: [], ...}----------
            if (epoch+1)%1 == 0 and epoch != 0:   #for every xx epoches, try eval
                print('@@@ epoch ------- ', epoch+1 , ' / ', self.epoch)
                ind = 0
                for id in self.nodes['node_id']:   #self.nodes['node_id']=self.look_back_list
                    self.vectors[id] = self.Embeddings[ind]
                    ind += 1
                #self.eval(vectors=self.vectors)
        print('please note that: the fianl embedding returned and its output file are not the best embedding!')
        print('for the best embeddings, please check which epoch got the best eval metric(s)......')


    def getEmbedding(self, type, nodes):
        if type == 'embed_layer':
            feed_dict = {self.train_data_id: nodes['node_id'], self.train_data_attr: nodes['node_attr']}
            Embedding = self.sess.run(self.embed_layer, feed_dict=feed_dict)
            return Embedding
        if type == 'out_embedding':
            Embedding = self.sess.run(self.weights['out_embeddings']) #sess.run to get embeddings from tf
            return Embedding  # nodes_number * (id_dim + attr_dim)
    
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

    def eval(self, vectors):
        #------nc task
        if self.task == 'nc' or self.task == 'nclp':
            print("Training nc classifier using {:.2f}% node labels...".format(self.nc_ratio*100))
            clf = ncClassifier(vectors=vectors, clf=LogisticRegression())   #use Logistic Regression as clf; we may choose SVM or more advanced ones
            clf.split_train_evaluate(self.X_nc_label, self.Y_nc_label, self.nc_ratio)
        #------lp task
        if self.task == 'lp':
            #X_test, Y_test = read_edge_label(args.label_file)  #enable this if you want to load your own lp testing data, see classfiy.py
            print("During embedding we have used {:.2f}% links and the remaining will be left for lp evaluation...".format(self.lp_ratio*100))
            clf = lpClassifier(vectors=vectors)     #similarity/distance metric as clf; basically, lp is a binary clf probelm
            clf.evaluate(self.X_test, self.Y_test)

