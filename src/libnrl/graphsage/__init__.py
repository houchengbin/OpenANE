''' global parameters for graphsage models
    tune these parameters here if needed
    if needed use: from libnrl.graphsage.__init__ import *
'''

# seed = 2018
# np.random.seed(seed)
# tf.set_random_seed(seed)
log_device_placement = False


# follow the original code by the paper author https://github.com/williamleif/GraphSAGE
# we follow the opt parameters given by papers GCN and graphSAGE
# note: citeseer+pubmed all follow the same parameters as cora, see their papers)
# tensorflow + Adam optimizer + Random weight init + row norm of attr
dim_1 = 64  # dim = dim1+dim2 = 128 for sage-mean and sage-gcn
dim_2 = 64
samples_1 = 25
samples_2 = 10

# key parameters during training
epochs = 100
learning_rate = 0.001  # search [0.01, 0.001, 0.0001, 0.00001]
dropout = 0.5
weight_decay = 5e-4
batch_size = 512  # if run out of memory, try to reduce them, default=512

# key parameters durning val
validate_batch_size = 256  # if run out of memory, try to reduce them, default=256
validate_iter = 5000
max_total_steps = 10**10
print_every = 50

# other parameters also follow the defaults https://github.com/williamleif/GraphSAGE
neg_sample_size = 20
identity_dim = 0
n2v_test_epochs = 1
random_context = False
model_size = 'small'
max_degree = 100
train_prefix = ''
base_log_dir = ''
base_log_dir = ''


'''
#core params..
flags.DEFINE_string('model', 'graphsage', 'model names. See README for possible values.')
flags.DEFINE_float('learning_rate', 0.00001, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'name of the object file that stores the training data. must be specified.')

# left to default values in main experiments
flags.DEFINE_integer('epochs', 1, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 100, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of users samples in layer 2')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('neg_sample_size', 20, 'number of negative samples')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_integer('n2v_test_epochs', 1, 'Number of new SGD epochs for n2v.')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', True, 'whether to save embeddings for all nodes after training')
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 50, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")
'''
