#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
from datetime import datetime
from operator import mul
from time import time

import cPickle
import importlib
import numpy as np
import sys
from neuralnet import NeuralNet
from deformer import Deformer, transform_flat_batch
from theano import shared, config

################################ HELPER FUNCTIONS ############################

def read_json_bz2(path2data):
    import bz2, json, contextlib
    with contextlib.closing(bz2.BZ2File(path2data, 'rb')) as f:
        return np.array(json.load(f))

def share(data, dtype=config.floatX):
    return shared(np.asarray(data, dtype), borrow=True)

class WrapOut:
    def __init__(self, use_file, name=''):
        self.name = name
        self.use_file = use_file
        if use_file:     self.stream = open(name, 'w', 1)
        else:            self.stream = sys.stdout

    def write(self, data):
        self.stream.write(data)

    def forceflush(self):
        if self.use_file:    
            self.stream.close()
            self.stream = open(self.name, 'a', 1)

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

################################### MAIN CODE ################################

if len(sys.argv) < 4:
    print('Usage:', sys.argv[0],
    ''' <xmatrix.bz2> <ymatrix.bz2> <params_file(s)>
    .bz2 files contain the samples and the output classes as generated
        by the gen_cnn_data.py script.
    params_file(s) :
        Parameters for the NeuralNet
        - params_file.py  : contains the initialization code
        - params_file.pkl : pickled file from a previous run (has wts too).
    ''')
    sys.exit()

##########################################  Import Parameters

prms_file_name = sys.argv[3]

if prms_file_name[-3:] == '.py':
    params = importlib.import_module(prms_file_name[:-3])
    layers = params.layers
    tr_prms = params.training_params
    allwts = None

elif prms_file_name[-4:] == '.pkl':
    with open(prms_file_name, 'rb') as f:
        params = cPickle.load(f)
    layers = params['layers']
    tr_prms = params['training_params']
    allwts = params['allwts']

else:
    raise NotImplementedError('Unknown file type for: ' + prms_file_name)

## Init SEED
if (not 'SEED' in tr_prms) or (tr_prms['SEED'] is None):
    tr_prms['SEED'] = np.random.randint(0, 1e7)

if len(sys.argv) > 4 and sys.argv[4] is '1':
    print("Printing output to {}.txt".format(tr_prms['SEED']))
    sys.stdout = WrapOut(True, str(tr_prms['SEED'])+'.txt')
else:
    sys.stdout = WrapOut(False)

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

print("\nLoading the data ...")
sys.stdout.forceflush()

data_x = read_json_bz2(sys.argv[1])
data_y = read_json_bz2(sys.argv[2])
print("X (samples, dimension) Size : {} {}KB\n"
      "Y (samples, dimension) Size : {} {}KB\n"
      "Y (min, max) : {} {}".format(data_x.shape, data_x.nbytes//1000, 
                                    data_y.shape, data_y.nbytes//1000,
                                    data_y.min(), data_y.max()))

corpus_sz, _ = data_x.shape
TRAIN = int(corpus_sz * tr_prms['TRAIN_ON_FRACTION'])

np_trin_x = data_x[:TRAIN,]
np_test_x = data_x[TRAIN:,]
np_trin_y = data_y[:TRAIN,].astype('int32')
np_test_y = data_y[TRAIN:,].astype('int32')

#trin_x = share(data_x[:TRAIN,])
test_x = share(data_x[TRAIN:,])
trin_y = share(data_y[:TRAIN,], 'int32')
test_y = share(data_y[TRAIN:,], 'int32')

print("\nInitializing the net ... ")  
nn = NeuralNet(layers, tr_prms, allwts)
print(nn)

def wrap_fn(f, dx, dy):
    def ret_fn(i):
        return f(dx[i*batch_sz:(i+1)*batch_sz],
                 dy[i*batch_sz:(i+1)*batch_sz])
    return ret_fn

print("\nCompiling ... ")
training_fn = nn.get_trin_model()
training_fn_wrap = wrap_fn(training_fn, np_trin_x, np_trin_y)
test_fn_tr = wrap_fn(nn.get_test_model(), np_trin_x, np_trin_y)
test_fn_te = wrap_fn(nn.get_test_model(), np_test_x, np_test_y)

tr_corpus_sz = TRAIN
te_corpus_sz = corpus_sz - TRAIN
batch_sz = layers[0][1]['batch_sz']
nEpochs = tr_prms['NUM_EPOCHS']
nTrBatches = tr_corpus_sz//batch_sz
nTeBatches = te_corpus_sz//batch_sz

############################################## MORE HELPERS 
def test_wrapper(nylist):
    sym_err, bit_err, n = 0., 0., 0
    for symdiff, bitdiff, _, _ in nylist:
        sym_err += symdiff
        bit_err += bitdiff
        n += 1
    return 100*sym_err/n, 100*bit_err/n

if nn.tr_layers[-1].kind == 'LOGIT':    aux_err_name = 'BitErr'
else:                                   aux_err_name = 'P(MLE)'

def get_test_indices(tot_samps, bth_samps=tr_prms['TEST_SAMP_SZ']):
    n_bths_each = int(bth_samps/batch_sz)
    n_bths_all  = int(tot_samps/batch_sz)
    cur = 0
    while True:
        yield [i%n_bths_all for i in range(cur, cur + n_bths_each)]
        cur = (cur + n_bths_each) % n_bths_all

test_indices = get_test_indices(te_corpus_sz)
trin_indices = get_test_indices(tr_corpus_sz)
pickle_file_name = str(tr_prms['SEED']) + '.pkl'

def do_test():
    test_err, aux_test_err = test_wrapper(test_fn_te(i) 
                                            for i in  test_indices.next())
    trin_err, aux_trin_err = test_wrapper(test_fn_tr(i) 
                                            for i in  trin_indices.next())
    print("{:5.2f}%  ({:5.2f}%)      {:5.2f}%  ({:5.2f}%)".format(
                trin_err, aux_trin_err, test_err, aux_test_err))
    sys.stdout.forceflush()

    with open(pickle_file_name, 'wb') as f:
        cPickle.dump(nn.get_init_params(), f, -1)

############################################ Deformer
wd = ht = layers[0][1]['img_sz']

############################################ Training Loop
import sharedmem

deform_time, train_time, test_time = 0, 0, 0
if not 'DFM_PRMS' in tr_prms:
    tr_prms['DEFORM'] = 'serial'
    tr_prms['DFM_PRMS'] = {'scale' : 64, 'sigma' : 8, 'cval'  : 0, 'ncpus' : 1},
df_prms = tr_prms['DFM_PRMS']

print("Training ...")
print("Epoch   Cost  Tr_Error Tr_{0}    Te_Error Te_{0}".format(aux_err_name))
for epoch in range(nEpochs):
    total_cost = 0

    if (tr_prms['DEFORM'] == 'parallel'):
        sh_np_trin_x = sharedmem.copy(np_trin_x)
        deformer = Deformer(sh_np_trin_x, batch_sz, (wd, ht), **df_prms)
        if epoch==0:
            print(deformer)
        for ibatch in deformer:
            output = training_fn(np_trin_x[ibatch*batch_sz:(ibatch+1)*batch_sz],
                                 np_trin_y[ibatch*batch_sz:(ibatch+1)*batch_sz])
            total_cost += output[0]
        del deformer
        del sh_np_trin_x

    else:
        for ibatch in range(nTrBatches):
            t = time()
            imgs = np_trin_x[ibatch*batch_sz:(ibatch+1)*batch_sz]
            if (tr_prms['DEFORM'] == 'serial'):
                imgs = transform_flat_batch(imgs, shape=(wd, ht), **df_prms)
            deform_time += time() - t
            t = time()
            output = training_fn(imgs, np_trin_y[ibatch*batch_sz:(ibatch+1)*batch_sz])
            train_time += time() - t
            total_cost += output[0]

    if epoch % tr_prms['EPOCHS_TO_TEST'] == 0:
        print("{:3d} {:>8.2f}".format(epoch, total_cost), end='    ')
        t = time()
        do_test()
        test_time = time() - t

    nn.inc_epoch_set_rate()

print('Optimization Complete!!!')
total_time = deform_time + train_time + test_time
deform_time = 100*deform_time/total_time
train_time = 100*train_time/total_time
test_time = 100*test_time/total_time
print('total_time : {:.3f}\n\t'
     'deform_time : {:.2f}%\n\t'
     'train_time  : {:.2f}%\n\t'
     'test_time   : {:.2f}%\n\t'.format(total_time, deform_time, train_time, test_time))
