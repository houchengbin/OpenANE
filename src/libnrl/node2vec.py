"""
NE method: DeepWalk and Node2Vec

modified by Chengbin Hou and Zeyu Dong 2018

originally from https://github.com/thunlp/OpenNE/blob/master/src/openne/node2vec.py
"""

from __future__ import print_function

import warnings

from gensim.models import Word2Vec

from . import walker

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


class Node2vec(object):
    def __init__(self, graph, path_length, num_paths, dim, p=1.0, q=1.0, dw=False, **kwargs):
        kwargs["workers"] = kwargs.get("workers", 1)
        if dw:
            kwargs["hs"] = 1
            p = 1.0
            q = 1.0

        self.graph = graph
        if dw:
            self.walker = walker.BasicWalker(graph, workers=kwargs["workers"])  # walker for deepwalk
        else:
            self.walker = walker.Walker(graph, p=p, q=q, workers=kwargs["workers"])  # walker for node2vec
            print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()
        sentences = self.walker.simulate_walks(num_walks=num_paths, walk_length=path_length)
        kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = kwargs.get("size", dim)
        kwargs["sg"] = 1

        self.size = kwargs["size"]
        print("Learning representation...")
        word2vec = Word2Vec(**kwargs)
        self.vectors = {}
        for word in graph.G.nodes():
            self.vectors[word] = word2vec.wv[word]
        del word2vec

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()
