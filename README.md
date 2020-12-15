# OpenANE: the first Open source framework specialized in Attributed Network Embedding (ANE)
We reproduce several ANE (Attributed Network Embedding) methods as well as PNE (Pure Network Embedding) methods in **one unified framework**, where they all share the same I/O, downstream tasks, etc. We start this project based on [OpenNE](https://github.com/thunlp/OpenNE) which mainly integrates PNE methods in one unified framework. 
<br> OpenANE not only integrates those PNE methods that consider pure structural information, but also provides the state-of-the-art ANE methods that consider both structural and attribute information during embedding.

Authors: Chengbin HOU chengbin.hou10@foxmail.com & Zeyu DONG zeyu.dong@foxmail.com 2018


## Motivation
In many real-world scenarios, a network often comes with node attributes such as paper metadata in a citation network, user profiles in a social network, and even node degrees in any plain networks. Unfortunately, PNE methods cannot make use of attribute information that may further improve the quality of node embeddings. 
<br> From engineering perspective, by offering more APIs to handle attribute information in graph.py and utils.py, OpenANE shall be easy to use for embedding an attributed network. Except attributed networks, OpenANE can also deal with plain networks by calling PNE methods, or by assigning node degrees as node attributes and then calling ANE methods. Therefore, to some extent, ANE methods can be regarded as the generalization of PNE methods.

## Methods
ANE methods: 
[ABRW](https://github.com/houchengbin/ABRW),
[ASNE](https://github.com/lizi-git/ASNE),
[AANE](https://github.com/xhuang31/AANE_Python),
[SAGE-Mean](https://github.com/williamleif/GraphSAGE),
[SAGE-GCN](https://github.com/williamleif/GraphSAGE),
[TADW](https://github.com/thunlp/OpenNE),
AttrComb,
AttrPure <br>
PNE methods:
[DeepWalk](https://github.com/thunlp/OpenNE),
[Node2Vec](https://github.com/thunlp/OpenNE),
[LINE](https://github.com/thunlp/OpenNE),
[GraRep](https://github.com/thunlp/OpenNE),
[others](https://github.com/thunlp/OpenNE)
<br> All methods in this framework are **unsupervised**, and so do not require any label during embedding phase.

For more details of each method, please have a look at our [paper](https://doi.org/10.1016/j.neucom.2020.05.080) or [preprint via ResearchGate link](https://www.researchgate.net/publication/341826514_RoSANE_Robust_and_Scalable_Attributed_Network_Embedding_for_Sparse_Networks). And if you find ABRW (namely RoSANE in the paper) or this frameworkis useful for your research, please consider citing it.
```
@article{hou2020RoSANE,
  title={RoSANE: Robust and Scalable Attributed Network Embedding for Sparse Networks},
  author={Hou, Chengbin and He, Shan and Tang, Ke},
  journal={Neurocomputing},
  year={2020},
  publisher={Elsevier},
  url={https://doi.org/10.1016/j.neucom.2020.05.080},
  doi={10.1016/j.neucom.2020.05.080},
}
```


## Usages
#### Requirements
```bash
cd OpenANE
pip install -r requirements.txt
```
Python 3.6.6 or above is required due to the new [`print(f' ')`](https://docs.python.org/3.6/reference/lexical_analysis.html#f-strings) feature
#### To obtain node embeddings as well as evaluate the quality
```bash
python src/main.py --method abrw --task lp_and_nc --emb-file emb/cora_abrw_emb --save-emb
```
#### To have an intuitive feeling in node embeddings
```bash
python src/vis.py --emb-file emb/cora_abrw_emb --label-file data/cora/cora_label.txt
```


## Testing (Cora)
### Parameter Settings
The default parameters for SAGE-GCN and SAGE-Mean are in *src/libnrl/graphsage/\__init\__.py*. And for other parameters:

| AANE_lamb | AANE_maxiter | AANE_rho | ABRW_alpha | ABRW_topk | ASNE_lamb | AttrComb_mode | GraRep_kstep | LINE_negative_ratio | LINE_order | Node2Vec_p | Node2Vec_q | TADW_lamb | TADW_maxiter | batch_size | dim | dropout | epochs | label_reserved | learning_rate | link_remove | number_walks | walk_length | weight_decay | window_size | workers |
|-----------|--------------|----------|------------|-----------|-----------|---------------|--------------|---------------------|------------|------------|------------|-----------|--------------|------------|-----|---------|--------|----------------|---------------|-------------|--------------|-------------|--------------|-------------|---------|
| 0.05      | 10           | 5        | 0.8        | 30        | 1         | concat        | 4            | 5                   | 3          | 0.5        | 0.5        | 0.2       | 10           | 128        | 128 | 0.5     | 100    | 0.7            | 0.001         | 0.1         | 10           | 80          | 0.0001       | 10          | 24      |


### Testing Results
#### Link Prediction (LP) and Node Classification (NC) tasks:
STEPS: Cora -> NE method -> node embeddings -> (downstream) LP/NC -> scores

| Method   | AUC (LP)   | Micro-F1 (NC) | Macro-F1 (NC) |
|----------|--------|----------|----------|
| aane     | 0.8081 | 0.7296   | 0.6941   |
| abrw     | 0.9376 | 0.8612   | 0.8523   |
| asne     | 0.7728 | 0.6052   | 0.5656   |
| attrcomb | 0.9053 | 0.8446   | 0.8318   |
| attrpure | 0.7993 | 0.7368   | 0.7082   |
| deepwalk | 0.8465 | 0.8147   | 0.8048   |
| grarep   | 0.8935 | 0.7632   | 0.7529   |
| line     | 0.6930 | 0.6130   | 0.5949   |
| node2vec | 0.7935 | 0.7938   | 0.7856   |
| sagegcn  | 0.8926 | 0.7964   | 0.7828   |
| sagemean | 0.8948 | 0.7899   | 0.7748   |
| tadw     | 0.8877 | 0.8442   | 0.8321   |

*We take the average of six runs. During embedding phase, 10% links are removed. During downstream phase, the removed 10% links and the equal number of non-existing links are used for LP testing; and 30% of labels are used for NC testing.

#### 2D Visualization task:
STEPS: Cora -> NE method -> node embeddings -> (downstream) PCA to 2D -> vis

<br> ![Cora vis](https://github.com/houchengbin/OpenANE/blob/master/log/vis.jpg) <br>

*The different colors indicate different ground truth labels.

## Other Datasets
More well-prepared (attributed) network datasets are available at [NetEmb-Datasets](https://github.com/houchengbin/NetEmb-Datasets)

### Your Own Dataset
```
*--------------- Structural Info (each row) --------------------*
adjlist: node_id1 node_id2 node_id3 ... (neighbors of node_id1)
or edgelist: node_id1 node_id2 weight (weight is optional)
*--------------- Attribute Info (each row) ---------------------*
node_id1 attr1 attr2 ...
*--------------- Label Info (each row) -------------------------*
node_id1 label1 label2 ...
```

### Parameter Tuning
For different dataset, one may need to search the optimal parameters instead of taking the default parameters.
For the meaning and suggestion of each parameter, please see main.py. 


## Contribution
We highly welcome and appreciate your contribution in fixing bugs, reproducing new ANE methods, etc. Please use the *pull requests* and your contribution will automatically appear in this project once accepted. We will add you to authors list, if your contribution is significant to this project.
