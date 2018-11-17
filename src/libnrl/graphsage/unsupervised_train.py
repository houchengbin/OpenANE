from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np

from libnrl.graphsage.models import SampleAndAggregate, SAGEInfo, Node2VecModel
from libnrl.graphsage.minibatch import EdgeMinibatchIterator
from libnrl.graphsage.neigh_samplers import UniformNeighborSampler
#from libnrl.graphsage.utils import load_data
from libnrl.graphsage.__init__ import *  #import default parameters


# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val = minibatch_iter.val_feed_dict(size)
    outs_val = sess.run([model.loss, model.ranks, model.mrr], 
                        feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

'''
def incremental_evaluate(sess, model, minibatch_iter, size):
    t_test = time.time()
    finished = False
    val_losses = []
    val_mrrs = []
    iter_num = 0
    while not finished:
        feed_dict_val, finished, _ = minibatch_iter.incremental_val_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.ranks, model.mrr], 
                            feed_dict=feed_dict_val)
        val_losses.append(outs_val[0])
        val_mrrs.append(outs_val[2])
    return np.mean(val_losses), np.mean(val_mrrs), (time.time() - t_test)
'''

def save_val_embeddings(sess, model, minibatch_iter, size, mod=""):
    val_embeddings = []
    finished = False
    seen = set([])  #this as set to store already seen emb-node id!
    nodes = []
    iter_num = 0
    name = "val"
    while not finished:
        feed_dict_val, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.mrr, model.outputs1], 
                            feed_dict=feed_dict_val)
        #ONLY SAVE FOR embeds1 because of planetoid
        for i, edge in enumerate(edges):
            if not edge[0] in seen:
                val_embeddings.append(outs_val[-1][i,:])
                nodes.append(edge[0])  #nodes: a list; has order
                seen.add(edge[0])  #seen: a set; NO order!!!
    #if not os.path.exists(out_dir):
    #    os.makedirs(out_dir)

    val_embeddings = np.vstack(val_embeddings)
    print(val_embeddings.shape)
    vectors = {}
    for i, embedding in enumerate(val_embeddings):
        vectors[nodes[i]] = embedding  #warning: seen: a set; nodes: a list
    return vectors

    '''  #if we want to save embs, modify the following code
    np.save(out_dir + name + mod + ".npy",  val_embeddings)
    with open(out_dir + name + mod + ".txt", "w") as fp:
        fp.write("\n".join(map(str,nodes)))
    '''

def construct_placeholders():
    # Define placeholders
    placeholders = {
        'batch1' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'batch2' : tf.placeholder(tf.int32, shape=(None), name='batch2'),
        # negative samples for all nodes in the batch
        'neg_samples': tf.placeholder(tf.int32, shape=(None,),
            name='neg_sample_size'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders


def train(train_data, test_data=None, model='graphsage_mean'):
    print('---------- the graphsage model we used: ', model)
    print('---------- parameters we sued: epochs, dim_1+dim_2, samples_1, samples_2, dropout, weight_decay, learning_rate, batch_size, normalize', 
            epochs, dim_1+dim_2, samples_1, samples_2, dropout, weight_decay, learning_rate, batch_size, normalize)
    G = train_data[0]
    features = train_data[1]  #note: features are in order of graph.look_up_list, since id_map = {k: v for v, k in enumerate(graph.look_back_list)}
    id_map = train_data[2]

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])
    
    random_context = False
    context_pairs = train_data[3] if random_context else None
    placeholders = construct_placeholders()
    minibatch = EdgeMinibatchIterator(G, 
            id_map,
            placeholders, batch_size=batch_size,
            max_degree=max_degree, 
            num_neg_samples=neg_sample_size,
            context_pairs = context_pairs)
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    if model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, samples_1, dim_1),
                            SAGEInfo("node", sampler, samples_2, dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     model_size=model_size,
                                     identity_dim = identity_dim,
                                     logging=True)
    elif model == 'gcn':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, samples_1, 2*dim_1),
                            SAGEInfo("node", sampler, samples_2, 2*dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="gcn",
                                     model_size=model_size,
                                     identity_dim = identity_dim,
                                     concat=False,
                                     logging=True)

    elif model == 'graphsage_seq':  #LSTM as stated in paper? very slow anyway...
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, samples_1, dim_1),
                            SAGEInfo("node", sampler, samples_2, dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     identity_dim = identity_dim,
                                     aggregator_type="seq",
                                     model_size=model_size,
                                     logging=True)

    elif model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, samples_1, dim_1),
                            SAGEInfo("node", sampler, samples_2, dim_2)]

        model = SampleAndAggregate(placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="maxpool",
                                     model_size=model_size,
                                     identity_dim = identity_dim,
                                     logging=True)
    elif model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, samples_1, dim_1),
                            SAGEInfo("node", sampler, samples_2, dim_2)]

        model = SampleAndAggregate(placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="meanpool",
                                     model_size=model_size,
                                     identity_dim = identity_dim,
                                     logging=True)

    elif model == 'n2v':
        model = Node2VecModel(placeholders, features.shape[0],
                                       minibatch.deg,
                                       #2x because graphsage uses concat
                                       nodevec_dim=2*dim_1,
                                       lr=learning_rate)
    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True
    
    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    #summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
     
    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
    
    # Train model
    
    train_shadow_mrr = None
    shadow_mrr = None

    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    for epoch in range(epochs): 
        minibatch.shuffle() 

        iter = 0
        epoch_val_costs.append(0)
        train_cost = 0
        train_mrr = 0
        train_shadow_mrr = 0
        val_cost = 0
        val_mrr = 0
        shadow_mrr = 0
        avg_time = 0
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: dropout})

            t = time.time()
            # Training step
            outs = sess.run([merged, model.opt_op, model.loss, model.ranks, model.aff_all, 
                    model.mrr, model.outputs1], feed_dict=feed_dict)
            train_cost = outs[2]
            train_mrr = outs[5]
            if train_shadow_mrr is None:
                train_shadow_mrr = train_mrr#
            else:
                train_shadow_mrr -= (1-0.99) * (train_shadow_mrr - train_mrr)

            if iter % validate_iter == 0:
                # Validation
                sess.run(val_adj_info.op)
                val_cost, ranks, val_mrr, duration  = evaluate(sess, model, minibatch, size=validate_batch_size)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost
            if shadow_mrr is None:
                shadow_mrr = val_mrr
            else:
                shadow_mrr -= (1-0.99) * (shadow_mrr - val_mrr)

            #if total_steps % print_every == 0:
                #summary_writer.add_summary(outs[0], total_steps)
    
            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            iter += 1
            total_steps += 1

            if total_steps > max_total_steps:
                break
        
        epoch += 1
        print("Epoch:", '%04d' % epoch, 
        "train_loss=", "{:.5f}".format(train_cost),
        "train_mrr=", "{:.5f}".format(train_mrr), 
        "train_mrr_ema=", "{:.5f}".format(train_shadow_mrr), # exponential moving average
        "val_loss=", "{:.5f}".format(val_cost),
        "val_mrr=", "{:.5f}".format(val_mrr), 
        "val_mrr_ema=", "{:.5f}".format(shadow_mrr), # exponential moving average
        "time=", "{:.5f}".format(avg_time))

        if total_steps > max_total_steps:
                break
    
    print("Optimization Finished!")

    sess.run(val_adj_info.op)
    #save_val_embeddings(sess, model, minibatch, validate_batch_size, log_dir())
    return save_val_embeddings(sess, model, minibatch, validate_batch_size)  #return embs


def graphsage_save_embeddings(self, filename): #to do...
    pass