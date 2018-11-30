'''
# author: Chengbin Hou @ SUSTech 2018 \n
# to tune parameters, refer to graphsage->__init__.py \n

# we provide utils to transform the orignal data into graphSAGE format \n
# the APIs are designed for unsupervised, \n
# for supervised way, plz refer and complete 'to do...' \n
# currently only support 'mean' and 'gcn' model \n
'''

import random

import networkx as nx
import numpy as np

from libnrl.graphsage.__init__ import *  # import default parameters


class graphSAGE(object):
    def __init__(self, graph, sage_model='mean', is_supervised=False):
        self.graph = graph
        self.normalize = True  # row normalization of node attributes
        self.num_walks = 50
        self.walk_len = 5

        self.add_train_val_test_to_G(test_perc=0.0, val_perc=0.1)  # if unsupervised, no test data
        train_data = self.tranform_data_for_graphsage()  # obtain graphSAGE required training data

        self.vectors = None
        if not is_supervised:
            from libnrl.graphsage import unsupervised_train
            self.vectors = unsupervised_train.train(train_data=train_data, test_data=None, model=sage_model)
        else:
            # to do...
            # from libnrl.graphsage import supervised_train
            # self.vectors = supervised_train.train()
            pass

    def add_train_val_test_to_G(self, test_perc=0.0, val_perc=0.1):
        ''' add if 'val' and/or 'test' to each node in G '''
        G = self.graph.G
        num_nodes = nx.number_of_nodes(G)
        test_ind = random.sample(range(0, num_nodes), int(num_nodes*test_perc))
        val_ind = random.sample(range(0, num_nodes), int(num_nodes*val_perc))
        for ind in range(0, num_nodes):
            id = self.graph.look_back_list[ind]
            if ind in test_ind:
                G.nodes[id]['test'] = True
                G.nodes[id]['val'] = False
            elif ind in val_ind:
                G.nodes[id]['test'] = False
                G.nodes[id]['val'] = True
            else:
                G.nodes[id]['test'] = False
                G.nodes[id]['val'] = False
        # Make sure the graph has edge train_removed annotations
        # (some datasets might already have this..)
        print("Loaded data.. now preprocessing..")
        for edge in G.edges():
            if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                    G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
                G[edge[0]][edge[1]]['train_removed'] = True
            else:
                G[edge[0]][edge[1]]['train_removed'] = False
        return G

    def tranform_data_for_graphsage(self):
        ''' OpenANE graph -> graphSAGE required format '''
        id_map = self.graph.look_up_dict
        G = self.graph.G
        feats = np.array([G.nodes[id]['attr'] for id in id_map.keys()])
        normalize = self.normalize
        if normalize and feats is not None:
            print("------------- row norm of node attributes ------------------", normalize)
            from sklearn.preprocessing import StandardScaler
            train_inds = [id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
            train_feats = feats[train_inds]
            scaler = StandardScaler()
            scaler.fit(train_feats)
            feats = scaler.transform(feats)
        # feats1 = nx.get_node_attributes(G,'test')
        # feats2 = nx.get_node_attributes(G,'val')
        walks = []
        walks = self.run_random_walks(num_walks=self.num_walks, walk_len=self.walk_len)
        class_map = 0  # to do... use sklearn to make class into binary form, no need for unsupervised...
        return G, feats, id_map, walks, class_map

    def run_random_walks(self, num_walks=50, walk_len=5):
        ''' generate random walks '''
        G = self.graph.G
        nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
        G = G.subgraph(nodes)
        pairs = []
        for count, node in enumerate(nodes):
            if G.degree(node) == 0:
                continue
            for i in range(num_walks):
                curr_node = node
                for j in range(walk_len):
                    if len(list(G.neighbors(curr_node))) == 0:  # isolated nodes! often appeared in real-world
                        break
                    next_node = random.choice(list(G.neighbors(curr_node)))  # changed due to compatibility
                    # next_node = random.choice(G.neighbors(curr_node))
                    # self co-occurrences are useless
                    if curr_node != node:
                        pairs.append((node, curr_node))
                    curr_node = next_node
            if count % 1000 == 0:
                print("Done walks for", count, "nodes")
        return pairs

    def save_embeddings(self, filename):
        ''' save embeddings to file '''
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        emb_dim = len(next(iter(self.vectors.values())))
        fout.write("{} {}\n".format(node_num, emb_dim))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()
