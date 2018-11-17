# -*- coding: utf-8 -*-

'''
#-----------------------------------------------------------------------------
# author: Chengbin Hou @ SUSTech 2018
# Email: Chengbin.Hou10@foxmail.com
# we provide utils to transform the orignal data into graphSAGE format
# you may easily use these APIs as what we demostrated in main.py of OpenANE
# the APIs are designed for unsupervised, for supervised way, plz complete 'label' to do codes...
#-----------------------------------------------------------------------------
'''
from networkx.readwrite import json_graph
import json
import random
import networkx as nx
import numpy as np
from libnrl.graphsage import unsupervised_train

def add_train_val_test_to_G(graph, test_perc=0.0, val_perc=0.1):  #due to unsupervised, we do not need test data
    G = graph.G  #take out nx G
    random.seed(2018)
    num_nodes = nx.number_of_nodes(G)
    test_ind = random.sample(range(0, num_nodes), int(num_nodes*test_perc))
    val_ind = random.sample(range(0, num_nodes), int(num_nodes*val_perc))
    for ind in range(0, num_nodes):
        id = graph.look_back_list[ind]
        if ind in test_ind:
            G.nodes[id]['test'] = True
            G.nodes[id]['val'] = False
        elif ind in val_ind:
            G.nodes[id]['test'] = False
            G.nodes[id]['val'] = True
        else:
            G.nodes[id]['test'] = False
            G.nodes[id]['val'] = False
    
    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False
    return G

def run_random_walks(G, num_walks=50, walk_len=5):
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(walk_len):
                if len(list(G.neighbors(curr_node))) == 0:  #isolated nodes! often appeared in real-world
                    break
                next_node = random.choice(list(G.neighbors(curr_node)))  #changed due to compatibility
                #next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

def tranform_data_for_graphsage(graph):
    G = add_train_val_test_to_G(graph)  #given OpenANE graph --> obtain graphSAGE graph
    #G_json = json_graph.node_link_data(G)  #train_data[0] in unsupervised_train.py

    id_map = graph.look_up_dict
    #conversion = lambda n : int(n)  # compatible with networkx >2.0
    #id_map = {conversion(k):int(v) for k,v in id_map.items()}  # due to graphSAGE requirement 

    feats = np.array([G.nodes[id]['feature'] for id in id_map.keys()])
    normalize = True  #have decleared in __init__.py
    if normalize and not feats is None:
        print("-------------row norm of node attributes/features------------------")
        from sklearn.preprocessing import StandardScaler
        train_inds = [id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
        train_feats = feats[train_inds]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    #feats1 = nx.get_node_attributes(G,'test')
    #feats2 = nx.get_node_attributes(G,'val')

    walks = []
    walks = run_random_walks(G, num_walks=50, walk_len=5) #use the defualt parameter in graphSAGE

    class_map = 0  #to do... use sklearn to make class into binary form, no need for unsupervised...
    return G, feats, id_map, walks, class_map

def graphsage_unsupervised_train(graph, graphsage_model = 'graphsage_mean'):
    train_data = tranform_data_for_graphsage(graph)
    #from unsupervised_train.py 
    vectors = unsupervised_train.train(train_data, test_data=None, model = graphsage_model)
    return vectors

'''
def save_embeddings(self, filename):
    fout = open(filename, 'w')
    node_num = len(self.vectors.keys())
    fout.write("{} {}\n".format(node_num, self.size))
    for node, vec in self.vectors.items():
        fout.write("{} {}\n".format(node,
                                    ' '.join([str(x) for x in vec])))
    fout.close()
'''