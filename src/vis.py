import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import os

def read_node_label(filename):
    with open(filename, 'r') as f:
        node_label = {}  #dict
        for l in f.readlines():
            vec = l.split()
            node_label[int(vec[0])] = str(vec[1:])
    return node_label

def read_node_emb(filename):
    with open(filename, 'r') as f:
        node_emb = {}  #dict
        next(f)  #except the head line: num_of_nodes, dim
        for l in f.readlines():
            vec = l.split()
            node_emb[int(vec[0])] = [float(i) for i in vec[1:]]
    return node_emb

# load the node label and saved embeddings
label_file = './data/cora/cora_label.txt'
emb_file = './emb/abrw.txt'
label_dict = read_node_label(label_file)
emb_dict = read_node_emb(emb_file)

if label_dict.keys() != emb_dict.keys():
    print('ERROR, node ids are not matched! Plz check again')
    exit(0)

#embeddings = np.array([i for i in emb_dict.values()], dtype=np.float32)
embeddings = np.array([emb_dict[i] for i in sorted(emb_dict.keys(), reverse=False)], dtype=np.float32)

labels = [label_dict[i] for i in sorted(label_dict.keys(), reverse=False)]


# save embeddings and labels
emb_df = pd.DataFrame(embeddings)
emb_df.to_csv('emb/log/embeddings.tsv', sep='\t', header=False, index=False)

lab_df = pd.Series(labels, name='label')
lab_df.to_frame().to_csv('emb/log/node_labels.tsv', header=False, index=False)

# save tf variable
embeddings_var = tf.Variable(embeddings, name='embeddings')
sess = tf.Session()

saver = tf.train.Saver([embeddings_var])
sess.run(embeddings_var.initializer)
saver.save(sess, os.path.join('emb/log', "model.ckpt"), 1)

# configure tf projector
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = 'embeddings'
embedding.metadata_path = 'node_labels.tsv'

projector.visualize_embeddings(tf.summary.FileWriter('emb/log'), config)

# type "tensorboard --logdir=emb/log" in CMD and have fun :)