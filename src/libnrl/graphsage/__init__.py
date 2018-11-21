''' global parameters for graphsage models
    tune these parameters here if needed
    if needed use: from libnrl.graphsage.__init__ import * 
'''

#seed = 2018
#np.random.seed(seed)
#tf.set_random_seed(seed)
log_device_placement = False


# follow the orignal code by the paper author https://github.com/williamleif/GraphSAGE
# we follow the opt parameters given by papers GCN and graphSAGE 
# note: citeseer+pubmed all follow the same parameters as cora, see their papers)
# tensorflow + Adam optimizer + Random weight init + row norm of attr
epochs = 100
dim_1 = 64      #dim = dim1+dim2 = 128 for sage-mean and sage-gcn
dim_2 = 64
learning_rate = 0.001
dropout = 0.5
weight_decay = 0.0001
batch_size = 128  #if run out of memory, try to reduce them, but we use the default e.g. 64, default=512
samples_1 = 25
samples_2 = 10


#other parameters that paper did not mentioned, but we also follow the defaults https://github.com/williamleif/GraphSAGE
model_size = 'small'
max_degree = 100
neg_sample_size = 20
random_context= True
validate_batch_size = 64  #if run out of memory, try to reduce them, but we use the default e.g. 64, default=256
validate_iter = 5000
max_total_steps = 10**10
n2v_test_epochs = 1
identity_dim = 0
train_prefix = ''
base_log_dir = ''
#print_every = 50