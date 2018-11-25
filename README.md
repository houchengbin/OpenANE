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

## Testing

### Parameter Setting

Currently, we use the default parameter

| AANE_lamb | AANE_maxiter | AANE_rho | ABRW_alpha | ABRW_topk | ASNE_lamb | AttrComb_mode | GraRep_kstep | LINE_negative_ratio | LINE_order | Node2Vec_p | Node2Vec_q | TADW_lamb | TADW_maxiter | batch_size | dim | dropout | epochs | label_reserved | learning_rate | link_remove | number_walks | walk_length | weight_decay | window_size | workers |
|-----------|--------------|----------|------------|-----------|-----------|---------------|--------------|---------------------|------------|------------|------------|-----------|--------------|------------|-----|---------|--------|----------------|---------------|-------------|--------------|-------------|--------------|-------------|---------|
| 0.05      | 10           | 5        | 0.8        | 30        | 1         | concat        | 4            | 5                   | 3          | 0.5        | 0.5        | 0.2       | 10           | 128        | 128 | 0.5     | 100    | 0.7            | 0.001         | 0.1         | 10           | 80          | 0.0001       | 10          | 24      |

### Testing Result

**citeseer**:

| method   | AUC    | Micro-F1 | Macro-F1 | Time     |
|----------|--------|----------|----------|----------|
| aane     | 0.8889 | 0.7067   | 0.6295   | 36.99    |
| abrw     | 0.9342 | 0.7287   | 0.6705   | 64.32    |
| asne     | 0.8293 | 0.5275   | 0.4736   | 70.89    |
| attrcomb | 0.8756 | 0.7077   | 0.6592   | 99.19    |
| attrpure | 0.8684 | 0.6922   | 0.6525   | 0.99     |
| deepwalk | 0.7203 | 0.5681   | 0.5205   | 93.14    |
| grarep   | 0.8501 | 0.5200   | 0.4656   | 16.48    |
| line     | 0.6340 | 0.3959   | 0.3503   | 242.59   |
| node2vec | 0.6588 | 0.5931   | 0.5493   | 27.60    |
| sagegcn  | 0.8953 | 0.6016   | 0.5247   | 444.50   |
| sagemean | 0.8772 | 0.6391   | 0.5606   | 371.74   |
| tadw     | 0.8984 | 0.7337   | 0.6866   | 14.39    |

**cora:**

| method   | AUC    | Micro-F1 | Macro-F1 | Time     |
|----------|--------|----------|----------|----------|
| aane     | 0.8158 | 0.7263   | 0.6904   | 26.48    |
| abrw     | 0.9290 | 0.8721   | 0.8603   | 48.94    |
| asne     | 0.7842 | 0.6076   | 0.5649   | 69.67    |
| attrcomb | 0.9111 | 0.8444   | 0.8284   | 60.32    |
| attrpure | 0.7857 | 0.7349   | 0.7039   | 0.49     |
| deepwalk | 0.8499 | 0.8100   | 0.8021   | 75.15    |
| grarep   | 0.8936 | 0.7669   | 0.7607   | 10.31    |
| line     | 0.6945 | 0.5873   | 0.5645   | 259.02   |
| node2vec | 0.7938 | 0.7977   | 0.7858   | 29.42    |
| sagegcn  | 0.8929 | 0.7780   | 0.7622   | 207.49   |
| sagemean | 0.8882 | 0.8057   | 0.7902   | 183.65   |
| tadw     | 0.9005 | 0.8383   | 0.8255   | 10.73    |

**mit:**

| method   | AUC    | Micro-F1 | Macro-F1 | Time     |
|----------|--------|----------|----------|----------|
| aane     | 0.6586 | 0.3742   | 0.0730   | 83.49    |
| abrw     | 0.9068 | 0.7981   | 0.2286   | 113.41   |
| asne     | 0.6596 | 0.3041   | 0.0681   | 901.18   |
| attrcomb | 0.8548 | 0.8016   | 0.2223   | 125.82   |
| attrpure | 0.6464 | 0.3497   | 0.0707   | 1.19     |
| deepwalk | 0.9190 | 0.7997   | 0.2295   | 173.70   |
| grarep   | 0.8983 | 0.7695   | 0.1901   | 51.55    |
| line     | 0.7836 | 0.7415   | 0.1857   | 6335.43  |
| node2vec | 0.9088 | 0.8085   | 0.2356   | 465.54   |
| sagegcn  | 0.8005 | 0.6057   | 0.1451   | 12462.35 |
| sagemean | 0.7525 | 0.5796   | 0.1279   | 10534.44 |

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
