'''
demo of using (attributed) Network Embedding methods;
STEP1: load data -->
STEP2: prepare data -->
STEP3: learn node embeddings -->
STEP4: downstream evaluations

python src/main.py --method abrw --save-emb False

by Chengbin Hou 2018 <chengbin.hou10@foxmail.com>
'''

import time
import random
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression #to do... 1) put it in downstream.py; and 2) try SVM...
from libnrl.classify import ncClassifier, lpClassifier, read_node_label
from libnrl.graph import *
from libnrl.utils import *
from libnrl import abrw #ANE method; Attributed Biased Random Walk
from libnrl import tadw #ANE method
from libnrl import aane #ANE method
from libnrl import asne #ANE method
from libnrl.gcn import gcnAPI #ANE method
from libnrl.graphsage import graphsageAPI #ANE method
from libnrl import attrcomb #ANE method
from libnrl import attrpure #NE method simply use svd or pca for dim reduction
from libnrl import node2vec #PNE method; including deepwalk and node2vec
from libnrl import line #PNE method
from libnrl.grarep import GraRep #PNE method
#from libnrl import TriDNR #to do... ANE method
#https://github.com/dfdazac/dgi #to do... ANE method


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    #-----------------------------------------------general settings--------------------------------------------------
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='graph/network format')
    parser.add_argument('--graph-file', default='data/cora/cora_adjlist.txt',
                        help='graph/network file')
    parser.add_argument('--attribute-file', default='data/cora/cora_attr.txt',
                        help='node attribute/feature file')
    parser.add_argument('--label-file', default='data/cora/cora_label.txt',
                        help='node label file') 
    parser.add_argument('--emb-file', default='emb/unnamed_node_embs.txt',
                        help='node embeddings file; suggest: data_method_dim_embs.txt')
    parser.add_argument('--save-emb', default=False, type=bool,
                        help='save emb to disk if True')       
    parser.add_argument('--dim', default=128, type=int,
                        help='node embeddings dimensions')
    parser.add_argument('--task', default='lp_and_nc', choices=['none', 'lp', 'nc', 'lp_and_nc'],
                        help='choices of downstream tasks: none, lp, nc, lp_and_nc')
    parser.add_argument('--link-remove', default=0.1, type=float, 
                        help='simulate randomly missing links if necessary; a ratio ranging [0.0, 1.0]')
    #parser.add_argument('--attr-remove', default=0.0, type=float, 
    #                    help='simulate randomly missing attributes if necessary; a ratio ranging [0.0, 1.0]')
    #parser.add_argument('--link-reserved', default=0.7, type=float, 
    #                    help='for lp task, train/test split, a ratio ranging [0.0, 1.0]')
    parser.add_argument('--label-reserved', default=0.7, type=float,
                        help='for nc task, train/test split, a ratio ranging [0.0, 1.0]')
    parser.add_argument('--directed', default=False, type=bool,
                        help='directed or undirected graph')
    parser.add_argument('--weighted', default=False, type=bool,
                        help='weighted or unweighted graph')
    #-------------------------------------------------method settings-----------------------------------------------------------
    parser.add_argument('--method', default='abrw', choices=['node2vec', 'deepwalk', 'line', 'gcn', 'grarep', 'tadw',
                                                            'abrw', 'asne', 'aane', 'attrpure', 'attrcomb', 'graphsage'],
                        help='choices of Network Embedding methods')
    parser.add_argument('--ABRW-topk', default=30, type=int,
                        help='select the most attr similar top k nodes of a node; ranging [0, # of nodes]') 
    parser.add_argument('--ABRW-alpha', default=0.8, type=float,
                        help='balance struc and attr info; ranging [0, 1]') 
    parser.add_argument('--TADW-lamb', default=0.2, type=float,
                        help='balance struc and attr info; ranging [0, inf]')       
    parser.add_argument('--AANE-lamb', default=0.05, type=float,
                        help='balance struc and attr info; ranging [0, inf]')
    parser.add_argument('--AANE-rho', default=5, type=float,
                        help='penalty parameter; ranging [0, inf]')
    parser.add_argument('--AANE-mode', default='comb', type=str, 
                        help='choices of mode: comb, pure')  
    parser.add_argument('--ASNE-lamb', default=1.0, type=float,
                        help='balance struc and attr info; ranging [0, inf]')
    parser.add_argument('--AttrComb-mode', default='concat', type=str,
                        help='choices of mode: concat, elementwise-mean, elementwise-max')
    parser.add_argument('--Node2Vec-p', default=0.5, type=float,
                        help='trade-off BFS and DFS; rid search [0.25; 0.50; 1; 2; 4]')             
    parser.add_argument('--Node2Vec-q', default=0.5, type=float,
                        help='trade-off BFS and DFS; rid search [0.25; 0.50; 1; 2; 4]')
    parser.add_argument('--GraRep-kstep', default=4, type=int,
                        help='use k-step transition probability matrix')
    parser.add_argument('--LINE-order', default=3, type=int, 
                        help='choices of the order(s), 1st order, 2nd order, 1st+2nd order')
    parser.add_argument('--LINE-no-auto-save', action='store_true',
                        help='no save the best embeddings when training LINE')
    parser.add_argument('--LINE-negative-ratio', default=5, type=int,
                        help='the negative ratio')
    #for walk based methods; some Word2Vec SkipGram parameters are not specified here
    parser.add_argument('--number-walks', default=10, type=int,
                        help='# of random walks of each node')
    parser.add_argument('--walk-length', default=80, type=int,
                        help='length of each random walk')
    parser.add_argument('--window-size', default=10, type=int, 
                        help='window size of skipgram model')
    parser.add_argument('--workers', default=24, type=int,
                        help='# of parallel processes.')
    #for deep learning based methods; parameters about layers and neurons used are not specified here
    parser.add_argument('--learning-rate', default=0.001, type=float,  
                        help='learning rate')        
    parser.add_argument('--batch-size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=100, type=int,
                        help='epochs')
    parser.add_argument('--dropout', default=0.5, type=float,  
                        help='dropout rate (1 - keep probability)')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight for L2 loss on embedding matrix')
    args = parser.parse_args()
    return args


def main(args):
    g = Graph() #see graph.py for commonly-used APIs and use g.G to access NetworkX APIs
    print('\nSummary of all settings: ', args)


    #---------------------------------------STEP1: load data-----------------------------------------------------
    print('\nSTEP1: start loading data......')
    t1 = time.time()
    #load graph structure info------
    if args.graph_format == 'adjlist':
        g.read_adjlist(path=args.graph_file, directed=args.directed)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(path=args.graph_file, weighted=args.weighted, directed=args.directed)
    #load node attribute info------
    is_ane = (args.method == 'abrw' or args.method == 'tadw' or args.method == 'gcn' or args.method == 'graphsage' or
                 args.method == 'attrpure' or args.method == 'attrcomb' or args.method == 'asne' or args.method == 'aane')
    if is_ane:
        assert args.attribute_file != ''
        g.read_node_attr(args.attribute_file)
    #load node label info------
    #to do... similar to attribute {'key_attribute': value}, label also loaded as {'key_label': value}
    t2 = time.time()
    print('STEP1: end loading data; time cost: {:.2f}s'.format(t2-t1))


    #---------------------------------------STEP2: prepare data----------------------------------------------------
    print('\nSTEP2: start preparing data for link pred task......')
    t1 = time.time()
    test_node_pairs=[]
    test_edge_labels=[]
    if args.task == 'lp' or args.task == 'lp_and_nc':
        edges_removed = g.remove_edge(ratio=args.link_remove)
        test_node_pairs, test_edge_labels = generate_edges_for_linkpred(graph=g, edges_removed=edges_removed, balance_ratio=1.0)
    t2 = time.time()
    print('STEP2: end preparing data; time cost: {:.2f}s'.format(t2-t1))


    #-----------------------------------STEP3: upstream embedding task-------------------------------------------------
    print('\nSTEP3: start learning embeddings......')
    print('the graph: ', args.graph_file, '\nthe # of nodes: ', g.get_num_nodes(), '\nthe # of edges used during embedding (edges maybe removed if lp task): ', g.get_num_edges(),
            '\nthe # of isolated nodes: ', g.get_num_isolates(), '\nis directed graph: ', g.get_isdirected(), '\nthe model used: ', args.method)
    t1 = time.time()
    model = None
    if args.method == 'abrw': 
        model = abrw.ABRW(graph=g, dim=args.dim, alpha=args.ABRW_alpha, topk=args.ABRW_topk, num_paths=args.number_walks,
                            path_length=args.walk_length, workers=args.workers, window=args.window_size)
    elif args.method == 'attrpure':
        model = attrpure.ATTRPURE(graph=g, dim=args.dim)
    elif args.method == 'attrcomb':
        model = attrcomb.ATTRCOMB(graph=g, dim=args.dim, comb_with='deepwalk',
                                     num_paths=args.number_walks, comb_method=args.AttrComb_mode)  #concat, elementwise-mean, elementwise-max
    elif args.method == 'asne':
        if args.task == 'nc':
            model = asne.ASNE(graph=g, dim=args.dim, alpha=args.ASNE_lamb, epoch=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size,
                             X_test=None, Y_test=None, task=args.task, nc_ratio=args.label_reserved, lp_ratio=args.link_reserved, label_file=args.label_file)
        else:
            model = asne.ASNE(graph=g, dim=args.dim, alpha=args.ASNE_lamb, epoch=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size,
                             X_test=test_node_pairs, Y_test=test_edge_labels, task=args.task, nc_ratio=args.label_reserved, lp_ratio=args.link_reserved, label_file=args.label_file)
    elif args.method == 'aane':
        model = aane.AANE(graph=g, dim=args.dim, lambd=args.AANE_lamb, mode=args.AANE_mode)
    elif args.method == 'tadw':
        model = tadw.TADW(graph=g, dim=args.dim, lamb=args.TADW_lamb)
    elif args.method == 'deepwalk':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length,
                                 num_paths=args.number_walks, dim=args.dim,
                                 workers=args.workers, window=args.window_size, dw=True)
    elif args.method == 'node2vec':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length, num_paths=args.number_walks, dim=args.dim,
                                 workers=args.workers, p=args.Node2Vec_p, q=args.Node2Vec_q, window=args.window_size)
    elif args.method == 'grarep':
        model = GraRep(graph=g, Kstep=args.GraRep_kstep, dim=args.dim)
    elif args.method == 'line':
        if args.label_file and not args.LINE_no_auto_save:
            model = line.LINE(g, epoch = args.epochs, rep_size=args.dim, order=args.LINE_order, 
                label_file=args.label_file, clf_ratio=args.label_reserved)
        else:
            model = line.LINE(g, epoch = args.epochs, rep_size=args.dim, order=args.LINE_order)
    elif args.method == 'graphsage':
        model = graphsageAPI.graphsage_unsupervised_train(graph=g, graphsage_model = 'graphsage_mean')  
        #we follow the default parameters, see __inti__.py in graphsage file
        #choices: graphsage_mean, gcn ......
        #model.save_embeddings(args.emb_file)  #to do...
    elif args.method == 'gcn':
        model = graphsageAPI.graphsage_unsupervised_train(graph=g, graphsage_model = 'gcn') #graphsage-gcn
    else:
        print('no method was found...')
        exit(0)
    '''
    elif args.method == 'gcn':   #OR use graphsage-gcn as in graphsage method...
        assert args.label_file != ''        #must have node label
        assert args.feature_file != ''      #different from previous ANE methods
        g.read_node_label(args.label_file)  #gcn is an end-to-end supervised ANE methoed
        model = gcnAPI.GCN(graph=g, dropout=args.dropout,
                            weight_decay=args.weight_decay, hidden1=args.hidden,
                            epochs=args.epochs, clf_ratio=args.label_reserved)
        #gcn does not have model.save_embeddings() func
    '''
    if args.save_emb:
        model.save_embeddings(args.emb_file + time.strftime(' %Y%m%d-%H%M%S', time.localtime()))
        print('Save node embeddings in file: ', args.emb_file)
    t2 = time.time()
    print('STEP3: end learning embeddings; time cost: {:.2f}s'.format(t2-t1))


    #---------------------------------------STEP4: downstream task-----------------------------------------------
    print('\nSTEP4: start evaluating ......: ')
    t1 = time.time()
    if args.method != 'semi_supervised_gcn':  #except semi-supervised methods, we will get emb first, and then eval emb
        vectors = 0
        if args.method == 'graphsage' or args.method == 'gcn':  #to do... run without this 'if'
            vectors = model                       
        else:
            vectors = model.vectors #for other methods....
        del model, g           
        #------lp task
        if args.task == 'lp' or args.task == 'lp_and_nc':
            #X_test_lp, Y_test_lp = read_edge_label(args.label_file)  #enable this if you want to load your own lp testing data, see classfiy.py
            print('Link Prediction task; the percentage of positive links for testing:' + 
                    '{:.2f} (by default, also generate equal # of negative links for testing)'.format(args.link_remove))
            clf = lpClassifier(vectors=vectors)     #similarity/distance metric as clf; basically, lp is a binary clf probelm
            clf.evaluate(test_node_pairs, test_edge_labels)
        #------nc task
        if args.task == 'nc' or args.task == 'lp_and_nc':
            X, Y = read_node_label(args.label_file)
            print('Node Classification task; the percentage of labels for testing: {:.2f}'.format(1-args.label_reserved))
            clf = ncClassifier(vectors=vectors, clf=LogisticRegression())   #use Logistic Regression as clf; we may choose SVM or more advanced ones
            clf.split_train_evaluate(X, Y, args.label_reserved)
    t2 = time.time()
    print('STEP4: end evaluating; time cost: {:.2f}s'.format(t2-t1))


if __name__ == '__main__':
    #random.seed(2018)
    #np.random.seed(2018)    
    main(parse_args())

