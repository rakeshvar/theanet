#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
from datetime import datetime
from operator import mul
from time import time

import ast
import cPickle
import importlib
import numpy as np
import sys
from neuralnet import NeuralNet
from theano import shared, config

# ############################### HELPER FUNCTIONS ############################


def read_json_bz2(path2data):
    import bz2, json, contextlib

    with contextlib.closing(bz2.BZ2File(path2data, 'rb')) as fdata:
        return np.array(json.load(fdata))


def share(data, dtype=config.floatX, borrow=True):
    return shared(np.asarray(data, dtype), borrow=borrow)


class WrapOut:
    def __init__(self, use_file, name=''):
        self.name = name
        self.use_file = use_file
        if use_file:
            self.stream = open(name, 'w', 1)
        else:
            self.stream = sys.stdout

    def write(self, data):
        self.stream.write(data)

    def forceflush(self):
        if self.use_file:
            self.stream.close()
            self.stream = open(self.name, 'a', 1)

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

# ################################## MAIN CODE ################################

if len(sys.argv) < 4:
    print('Usage:', sys.argv[0],
          ''' <x.bz2> <y.bz2> <params_file(s)> [auxillary.bz2] [redirect=0]
    .bz2 files contain the samples and the output classes as generated
        by the gen_cnn_data.py script (or the like).
    params_file(s) :
        Parameters for the NeuralNet
        - params_file.py  : contains the initialization code
        - params_file.pkl : pickled file from a previous run (has wts too).
    out2file:
    	1 - redirect stdout to a <SEED>.txt file
    ''')
    sys.exit()

# #########################################  Import Parameters

prms_file_name = sys.argv[3]

with open(prms_file_name, 'rb') as f:
    if prms_file_name.endswith('.ast'):
        params = ast.literal_eval(f.read())

    elif prms_file_name.endswith('.pkl'):
        params = cPickle.load(f)

    else:
        raise NotImplementedError('Unknown file type for: ' + prms_file_name)

    layers = params['layers']
    tr_prms = params['training_params']
    try:
        allwts = params['allwts']
    except KeyError:
        allwts = None

## Init SEED
if (not 'SEED' in tr_prms) or (tr_prms['SEED'] is None):
    tr_prms['SEED'] = np.random.randint(0, 1e6)

if sys.argv[-1] is '1':
    print("Printing output to {}.txt".format(tr_prms['SEED']))
    sys.stdout = WrapOut(True, str(tr_prms['SEED']) + '.txt')
else:
    sys.stdout = WrapOut(False)


##########################################  Print Parameters

print(' '.join(sys.argv))
print('Time   : ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('Device : {} ({})'.format(config.device, config.floatX))

## Print Layers Info
for layer in layers:
    print('{} : '.format(layer[0]))
    for key in layer[1]:
        print('\t{} : \t{}'.format(key, layer[1][key]))

## Print sizes of weights
if allwts:
    shp = [[w.shape for w in ww] for ww in allwts]
    typ = [[w.dtype for w in ww] for ww in allwts]
    nwts = sum([sum([reduce(mul, w) for w in ww]) for ww in shp])
    print('\nTotal Number of Weights : {}'.format(nwts))
    for sh, tp in zip(shp, typ):
        print('                          |- {}'.format(zip(sh, tp)))

## Print Training Parameters
print('\nTraing Parameters: ')
for key, val in tr_prms.items():
    print('\t{} : \t{}'.format(key, val))

##########################################  Load Data

print("\nInitializing the net ... ")
nn = NeuralNet(layers, tr_prms, allwts)
print(nn)

print("\nLoading the data ...")
sys.stdout.forceflush()

batch_sz = tr_prms['BATCH_SZ']
img_sz = layers[0][1]['img_sz']
data_x = read_json_bz2(sys.argv[1])
data_y = read_json_bz2(sys.argv[2])

corpus_sz = data_x.shape[0]
data_x = data_x.reshape((corpus_sz, img_sz, img_sz))

print("X (samples, dimension) Size : {} {}KB\n"
      "Y (samples, dimension) Size : {} {}KB\n"
      "Y (min, max) : {} {}".format(data_x.shape, data_x.nbytes // 1000,
                                    data_y.shape, data_y.nbytes // 1000,
                                    data_y.min(), data_y.max()))

n_train = int(corpus_sz * tr_prms['TRAIN_ON_FRACTION'])

trin_x = share(data_x[:n_train, ])
test_x = share(data_x[n_train:, ])
trin_y = share(data_y[:n_train, ], 'int32')
test_y = share(data_y[n_train:, ], 'int32')

if len(sys.argv) > 4:
    data_aux = read_json_bz2(sys.argv[4])
    data_aux = data_aux.reshape((corpus_sz, 2, 2))
    trin_aux = share(data_aux[:n_train, ])
    test_aux = share(data_aux[n_train:, ])
else:
    trin_aux, test_aux = None, None

print("\nCompiling ... ")
training_fn = nn.get_trin_model(trin_x, trin_y, trin_aux)
test_fn_tr = nn.get_test_model(trin_x, trin_y, trin_aux)
test_fn_te = nn.get_test_model(test_x, test_y, test_aux)

tr_corpus_sz = n_train
te_corpus_sz = corpus_sz - n_train
nEpochs = tr_prms['NUM_EPOCHS']
nTrBatches = tr_corpus_sz // batch_sz
nTeBatches = te_corpus_sz // batch_sz

############################################## MORE HELPERS 


def test_wrapper(nylist):
    sym_err, bit_err, n = 0., 0., 0
    for symdiff, bitdiff in nylist:
        sym_err += symdiff
        bit_err += bitdiff
        n += 1
    return 100 * sym_err / n, 100 * bit_err / n


if nn.tr_layers[-1].kind == 'LOGIT':
    aux_err_name = 'BitErr'
else:
    aux_err_name = 'P(MLE)'


def get_test_indices(tot_samps, bth_samps=tr_prms['TEST_SAMP_SZ']):
    n_bths_each = int(bth_samps / batch_sz)
    n_bths_all = int(tot_samps / batch_sz)
    cur = 0
    while True:
        yield [i % n_bths_all for i in range(cur, cur + n_bths_each)]
        cur = (cur + n_bths_each) % n_bths_all


test_indices = get_test_indices(te_corpus_sz)
trin_indices = get_test_indices(tr_corpus_sz)
pickle_file_name = str(tr_prms['SEED']) + '.pkl'


def do_test():
    test_err, aux_test_err = test_wrapper(test_fn_te(i)
                                          for i in test_indices.next())
    trin_err, aux_trin_err = test_wrapper(test_fn_tr(i)
                                          for i in trin_indices.next())
    print("{:5.2f}%  ({:5.2f}%)      {:5.2f}%  ({:5.2f}%)".format(
        trin_err, aux_trin_err, test_err, aux_test_err))
    sys.stdout.forceflush()

    with open(pickle_file_name, 'wb') as pkl_file:
        cPickle.dump(nn.get_init_params(), pkl_file, -1)


############################################ Training Loop

print("Training ...")
print("Epoch   Cost  Tr_Error Tr_{0}    Te_Error Te_{0}".format(aux_err_name))
for epoch in range(nEpochs):
    total_cost = 0

    for ibatch in range(nTrBatches):
        output = training_fn(ibatch)
        total_cost += output[0]

    if epoch % tr_prms['EPOCHS_TO_TEST'] == 0:
        print("{:3d} {:>8.2f}".format(nn.get_epoch(), total_cost), end='    ')
        t = time()
        do_test()
        test_time = time() - t

    nn.inc_epoch_set_rate()

print('Optimization Complete!!!')
