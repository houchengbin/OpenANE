# OpenANE: The first Open source framework specialized in Attributed Network Embedding (ANE)
We reproduce several ANE (Attributed Network Embedding) as well as PNE (Pure Network Embedding) methods in one unified framework, where they all share the same I/O and downstream tasks. We start this project based on [OpenNE](https://github.com/thunlp/OpenNE) that mainly integrates PNE methods under one unified framework. 
<br> OpenANE not only integrates those PNE methods that consider pure structural information, but also provides the state-of-the-art ANE methods that consider both structural and attribute information during embedding.

Authors: Chengbin HOU chengbin.hou10@foxmail.com & Zeyu DONG 2018


## Motivation
In many real-world scenarios, a network often comes with node attributes such as paper metadata in a citation network, user profiles in a social network, and even node degrees in any pure networks. Unfortunately, PNE methods cannot make use of attribute information that may further improve the quality of node embeddings. 
<br> From engineering perspective, by offering more APIs to handle attribute information in graph.py and utils.py, OpenANE shall be very easy to use for embedding an attributed network. Except attributed networks, OpenANE can also deal with pure networks by calling PNE methods, or by assigning node degrees as node attributes and then calling ANE methods. Therefore, to some extent, ANE methods can be regarded as a generalization of PNE methods.

## Methods
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
AttrComb
<br> Note: all NE methods in this framework are unsupervised, and so does not require any label during embedding phase.

**For more details of each method, please have a look at our paper https://arxiv.org/abs/1811.11728**
<br> And if you find ABRW or this framework is useful for your research, please consider citing it.


## Usages
#### Requirements
```bash
pip install -r requirements.txt
```
Python 3.6.6 or above is required due to the new [*print(f' ')*](https://docs.python.org/3.6/reference/lexical_analysis.html#f-strings) feature
#### To obtain node embeddings as well as evaluate the quality
```bash
python src/main.py --method abrw --emb-file emb/cora_abrw_emb --save-emb --task lp_and_nc
```
#### To have an intuitive feeling in node embeddings
```bash
python src/vis.py --emb-file emb/cora_abrw_emb --label-file data/cora/cora_label.txt
```


## Testing (Cora)
### Parameter Settings
In this testing, we use the default parameters

| AANE_lamb | AANE_maxiter | AANE_rho | ABRW_alpha | ABRW_topk | ASNE_lamb | AttrComb_mode | GraRep_kstep | LINE_negative_ratio | LINE_order | Node2Vec_p | Node2Vec_q | TADW_lamb | TADW_maxiter | batch_size | dim | dropout | epochs | label_reserved | learning_rate | link_remove | number_walks | walk_length | weight_decay | window_size | workers |
|-----------|--------------|----------|------------|-----------|-----------|---------------|--------------|---------------------|------------|------------|------------|-----------|--------------|------------|-----|---------|--------|----------------|---------------|-------------|--------------|-------------|--------------|-------------|---------|
| 0.05      | 10           | 5        | 0.8        | 30        | 1         | concat        | 4            | 5                   | 3          | 0.5        | 0.5        | 0.2       | 10           | 128        | 128 | 0.5     | 100    | 0.7            | 0.001         | 0.1         | 10           | 80          | 0.0001       | 10          | 24      |

### Testing Results
#### Link Prediction and Node Classification tasks:

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

#### 2D Visualization task
<br> ![Cora vis](https://github.com/houchengbin/OpenANE/blob/master/log/vis.jpg) <br>
<br> Steps: Cora -> NE method -> node embeddings -> PCA -> 2D vis
<br> The different colors indicate different ground truth labels.

## Other Datasets
More well-prepared (attributed) network datasets are available at [NetEmb-Datasets](https://github.com/houchengbin/NetEmb-Datasets)

### Your Own Dataset
    *--------------- Structural Info (each row) --------------------*
    adjlist: node_id1 node_id2 node_id3 ... (neighbors of node_id1)
    or edgelist: node_id1 node_id2 weight (weight is optional)
    *--------------- Attribute Info (each row) ---------------------*
    node_id1 attr1 attr2 ...
    *--------------- Label Info (each row) -------------------------*
    node_id1 label1 label2 ...

### Parameter Tuning
For different dataset, one may need to search the optimal parameters instead of taking the default parameters.
For the meaning and suggestion of each parameter, please see main.py. 


## Contribution
We highly welcome and appreciate your contribution on fixing bugs, reproducing new ANE methods, etc. And we hope this OpenANE framework would become influential on both academic research and industrial usage.


## References
todo...
