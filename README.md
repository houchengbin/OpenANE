# OpenANE: The first open source framework specialized in Attributed Network Embedding (ANE)
We reproduce several ANE (Attributed Network Embedding) as well as PNE (Pure Network Embedding) methods in one framework, where they all share the same I/O and downstream tasks. We start this project based on the excellent [OpenNE](https://github.com/thunlp/OpenNE) project that integrates several PNE methods under the same framework. However, OpenANE not only integrates those PNE methods from OpenNE, but also provides the state-of-the-art ANE methods that consider both structural and attribute information during embedding.

authors: Chengbin HOU (chengbin.hou10@foxmail.com) & Zeyu DONG 2018

## Motivation
In many real-world scenarios, a network often comes with node attributes such as the paper's title in a citation network and user profiles in a social network. PNE methods that only consider structural information cannot make use of attribute information which may further improve the quality of node embedding. 

From engineering perspective, by offering more APIs to handle attribute information in graph.py and utils.py, OpenANE shall be very easy to use for embedding an attributed network. Of course, OpenANE can also deal with pure network: 1) by calling PNE methods; and 2) by assigning ones as the attribute for all nodes and then calling ANE methods (but some ANE methods may fail).

## Methods (todo... Chengbin)
[ABRW](https://github.com/houchengbin/ABRW),
[SAGE-GCN](https://github.com/williamleif/GraphSAGE),
[SAGE-Mean](https://github.com/williamleif/GraphSAGE),
[ASNE](https://github.com/lizi-git/ASNE),
[TADW](https://github.com/thunlp/OpenNE),
[AANE](https://github.com/xhuang31/AANE_Python),
[DeepWalk](https://github.com/thunlp/OpenNE),
[Node2Vec](https://github.com/thunlp/OpenNE),
[LINE](https://github.com/thunlp/OpenNE),
[GraRep](https://github.com/thunlp/OpenNE),
AttrPure,
AttrComb,

## Requirements (todo... Zeyu; double check)
pip install -r requirements.txt

## Usages
#### To obtain your node embeddings and evaluate them by classification downstream tasks
python src/main.py --method abrw --emb-file cora_abrw_emb --save-emb
#### To have an intuitive feeling of node embeddings (todo... Zeyu if possible; need tf installed)
python src/viz.py --emb-file cora_abrw_emb --label-file data/cora_label

## Parameters
#### the meaning of each parameter
please see main.py
#### searching optimal value of parameter (todo... Chengbin)
ABRW
SAGE-GCN

## Testing (todo... Zeyu)
Currently, we use the default parameter....

.... summary of parameters ....

.... table --- results ....


## Datasets (todo...Chengbin)
We provide Cora for ... and other datasets e.g. Facebook_MIT, refer to [NetEmb-Datasets](https://github.com/houchengbin/NetEmb-datasets)

### Your own dataset?
#### FILE for structural information (each row):
adjlist: node_id1 node_id2 node_id3 -> (the edges between (id1, id2) and (id1, id3)) 

OR edgelist: node_id1 node_id2 weight(optional) -> one edge (id1, id2)
#### FILE for attribute information (each row):
node_id1 attr1 attr2 ... attrM

#### FILE for label (each row):
node_id1 label(s)

## Want to contribute?
We highly welcome and appreciate your contributions on fixing bugs, reproducing new ANE methods, etc. And together, we hope this OpenANE framework would become influential on both academic research and industrial usage.

## Recommended References (todo... Chengbin)
