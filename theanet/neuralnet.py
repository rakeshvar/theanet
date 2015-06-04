from __future__ import print_function
import numpy as np
import theano
import theano.tensor as tt

from . import layer
from .layer import InputLayer, ElasticLayer
from .layer import ConvLayer, PoolLayer
from .layer import HiddenLayer, AuxConcatLayer
from .layer import SoftmaxLayer, CenteredOutLayer, SoftAuxLayer

# theano.config.optimizer = 'fast_compile'
# theano.config.exception_verbosity = "high"

############################ Helper Functions #################################
from functools import reduce
from operator import mul


def get_layers_info(layers):
    string = ""
    for lyr in layers:
        string += '\n{} : '.format(lyr[0])
        for key in lyr[1]:
            string += '\n\t{} : \t{}'.format(key, lyr[1][key])

    return string


def get_wts_info(wts, detailed=False):
    string, n_wts = "", 0
    for l, ww in enumerate(wts):
        string += "\nLayer {}:".format(l)
        for w in ww:
            n_ww = reduce(mul, w.shape)
            n_wts += n_ww
            string += '\n\t {} {} ❲{}❳'.format(w.shape, w.dtype, n_ww)
            if detailed:
                string += " ❲{:.2e}, {:.2e}, {:.2e}❳".format(
                    w.min(), w.mean(), w.max())

    string += '\n\nTotal Number of Weights : {:,}'.format(n_wts)
    return string


def get_training_params_info(training_params):
    string = "Training Parameters:"
    for key in sorted(training_params.keys()):
        string += '\n\t{} : \t{}'.format(key, training_params[key])

    return string


###############################################################################
#                           The Neural Network
###############################################################################


class NeuralNet():
    def __init__(self, layers, training_params, allwts=None):

        # Either we have a random seed or the WTS for each layer from a 
        # previously trained NeuralNet
        if allwts is None:
            rand_gen = np.random.RandomState(training_params['SEED'])
        else:
            rand_gen = None

        # Symbolic variables for the data
        x = tt.tensor3('x')
        test_x = tt.tensor3('test_x')
        y = tt.ivector('y')

        tr_layers = []
        te_layers = []

        # Input Layer
        assert layers[0][0] in ('InputLayer', 'ElasticLayer'), \
            "First layer needs to be Input or Elastic Distorition Layer"
        batch_sz = training_params['BATCH_SZ']

        ilayer = 0
        if layers[0][0] == 'InputLayer':
            tr_layers.append(InputLayer(x, **layers[0][1]))

        elif layers[0][0] == 'ElasticLayer':
            tr_layers.append(ElasticLayer(x, **layers[0][1]))

        te_layers.append(tr_layers[0].TestVersion(test_x))
        ilayer += 1

        # ConvPool Layers
        while layers[ilayer][0] in ('ConvLayer', 'PoolLayer'):
            prev_tr_layer = tr_layers[ilayer-1]
            prev_te_layer = te_layers[ilayer-1]

            tr_inpt = prev_tr_layer.output
            te_inpt = prev_te_layer.output

            if type(prev_tr_layer) in (InputLayer, ElasticLayer):
                img_sz = prev_tr_layer.out_sz
                tr_inpt = tr_inpt.reshape((batch_sz, 1, img_sz, img_sz))
                te_inpt = te_inpt.reshape((batch_sz, 1, img_sz, img_sz))

            wts = allwts[ilayer] if allwts else None

            if layers[ilayer][0] == "ConvLayer":
                curr_layer = ConvLayer(tr_inpt, wts, rand_gen, batch_sz,
                                       prev_tr_layer.num_maps,
                                       prev_tr_layer.out_sz,
                                       **layers[ilayer][1])
            else:
                curr_layer = PoolLayer(tr_inpt,
                                       prev_tr_layer.num_maps,
                                       prev_tr_layer.out_sz,
                                       **layers[ilayer][1])

            tr_layers.append(curr_layer)
            te_layers.append(curr_layer.TestVersion(te_inpt))
            ilayer += 1

        # Flatten Output of Last ConvPool Layers
        prev_tr_layer = tr_layers[ilayer - 1]
        prev_te_layer = te_layers[ilayer - 1]
        prev_tr_layer.output = prev_tr_layer.output.flatten(2)
        prev_te_layer.output = prev_te_layer.output.flatten(2)

        #  Hidden Layers
        while layers[ilayer][0] in ('AuxConcatLayer', 'HiddenLayer'):
            prev_tr_layer = tr_layers[ilayer - 1]
            prev_te_layer = te_layers[ilayer - 1]

            wts = allwts[ilayer] if allwts else None

            curr_layer_type = getattr(layer, layers[ilayer][0])
            curr_layer = curr_layer_type(prev_tr_layer.output, wts, rand_gen,
                                         prev_tr_layer.n_out,
                                         **layers[ilayer][1])

            tr_layers.append(curr_layer)
            te_layers.append(curr_layer.TestVersion(prev_te_layer.output))
            ilayer += 1

        # Output layer
        #   |- CenteredOutLayer 
        #       |- Can be LOGIT or RBF
        #       |- Needs Wts for hidden layer (can be generated randomly as nPrevLayerUnits x nFeatures)
        #       |- Needs CENTERS as a matrix or as a tuple (nClassesxnFeatures) to convert from Features to Class probs
        #   |- SoftmaxLayer
        #       |- Needs Wts for hidden layer (can be generated randomly as nPrevLayerUnits x nClasses)
        #       |- Needs CENTERS as nClasses
        assert layers[ilayer][0] in ('SoftmaxLayer',
                                     'SoftAuxLayer',
                                     'SVMLayer',
                                     'CenteredOutLayer'), \
            "Hidden Layers need to be followed by OutputLayer"

        wts = allwts[ilayer] if allwts else None
        prev_tr_layer = tr_layers[ilayer - 1]
        prev_te_layer = te_layers[ilayer - 1]

        if layers[ilayer][0][:3] in ('Sof', 'SVM'):
            curr_layer_type = getattr(layer, layers[ilayer][0])
            curr_layer = curr_layer_type(prev_tr_layer.output,
                                         wts, rand_gen,
                                         prev_tr_layer.n_out,
                                         **layers[ilayer][1])

        elif layers[ilayer][0] == 'CenteredOutLayer':
            try:
                centers = layers[ilayer][1].pop('centers')
            except KeyError:
                centers = None

            if allwts and len(allwts) == 3:
                wts = allwts[:2]
                centers = allwts[3]

            curr_layer = CenteredOutLayer(prev_tr_layer.output, wts, centers,
                                          rand_gen, prev_tr_layer.n_out,
                                          **layers[ilayer][1])
        else:
            raise NotImplementedError("Unknown Layer of type : ",
                                      layers[ilayer][0])

        tr_layers.append(curr_layer)
        te_layers.append(curr_layer.TestVersion(prev_te_layer.output))
        ilayer += 1

        for tr_layer, te_layer in zip(tr_layers, te_layers):
            if type(tr_layer) in (AuxConcatLayer, SoftAuxLayer):
                assert not hasattr(self, 'aux_inpt_tr'), "Multiple Aux Inputs"
                self.aux_inpt_tr = tr_layer.aux_inpt
                self.aux_inpt_te = te_layer.aux_inpt

        # Store variables that need to be persistent
        self.tr_layers = tr_layers
        self.te_layers = te_layers
        self.num_layers = ilayer
        self.tr_prms = training_params
        self.layers = layers
        self.x = x
        self.y = y
        self.test_x = test_x
        self.batch_sz = batch_sz

        # Set Epoch and learning rate
        if 'CUR_EPOCH' not in training_params:
            training_params['CUR_EPOCH'] = 0
        self.cur_learn_rate = theano.shared(np.cast[theano.config.floatX](0.0))
        self.set_rate()

    def get_trin_model(self, x_data, y_data, aux_data=None):
        # cost, training params, gradients, updates to those params
        print('Compiling training function...')

        cost = self.tr_layers[-1].cost(self.y)
        for lyr in self.tr_layers:
            cost += lyr.get_wtcost()

        updates = []
        for lyr in self.tr_layers:
            updates += lyr.get_updates(cost, self.cur_learn_rate)

        indx = tt.lscalar('training batch index')
        bth_sz = self.tr_prms['BATCH_SZ']

        givens = {
            self.x: x_data[indx*bth_sz:(indx+1)*bth_sz],
            self.y: y_data[indx*bth_sz:(indx+1)*bth_sz], }

        if hasattr(self, 'aux_inpt_tr'):
            assert aux_data is not None, "Auxillary data not supplied"
            givens[self.aux_inpt_tr] = aux_data[indx*bth_sz:(indx+1)*bth_sz]

        return theano.function([indx],
                               [cost,
                                self.tr_layers[-1].features,
                                self.tr_layers[-1].logprob],
                               updates=updates,
                               givens=givens)

    def get_test_model(self, x_data, y_data, aux_data=None, preds_feats=False):
        print('Compiling testing function... ')
        idx = tt.lscalar('test batch index')
        bth_sz = self.tr_prms['BATCH_SZ']

        givens = {
            self.test_x: x_data[idx*bth_sz:(idx+1)*bth_sz],
            self.y: y_data[idx*bth_sz:(idx+1)*bth_sz]}

        if hasattr(self, 'aux_inpt_te'):
            assert aux_data is not None, "Auxillary data not supplied"
            givens[self.aux_inpt_te] = \
                aux_data[idx*bth_sz:(idx+1)*bth_sz]

        outputs = self.te_layers[-1].sym_and_oth_err_rate(self.y)
        if preds_feats:
            outputs += self.te_layers[-1].features_and_predictions()

        return theano.function([idx],
                               outputs,
                               givens=givens)

    def takes_aux(self):
        return hasattr(self, 'aux_inpt_te')

    def get_data_test_model(self, go_nuts=False):
        print('Compiling full test function...')
        inputs = [self.test_x]
        if self.takes_aux():
            inputs += [self.aux_inpt_te]

        outputs = list(self.te_layers[-1].features_and_predictions())
        if go_nuts:
            print("going nuts...")
            for lyr in self.te_layers:
                outputs.append(lyr.output)

        return theano.function(inputs, outputs)

    def get_init_params(self):
        return {"layers": self.layers,
                "training_params": self.tr_prms,
                "allwts": [l.get_wts() for l in self.tr_layers]}

    def set_rate(self):
        self.cur_learn_rate.set_value(
            self.tr_prms['INIT_LEARNING_RATE'] /
                (1 + self.tr_prms['CUR_EPOCH'] /
                         self.tr_prms['EPOCHS_TO_HALF_RATE']))

    def inc_epoch_set_rate(self):
        self.tr_prms['CUR_EPOCH'] += 1
        self.set_rate()

    def get_epoch(self):
        return self.tr_prms['CUR_EPOCH']

    def __str__(self):
        prmstr = '; '.join([', '.join([str(prm) for prm in lyr.params])
                            for lyr in self.tr_layers])
        return \
            '\nTrain Layers\n\t' + \
            '\n\t'.join([str(l) for l in self.tr_layers]) + \
            '\nTest Layers\n\t' + \
            '\n\t'.join([str(l) for l in self.te_layers]) + \
            '\nParams ' + prmstr

    def get_layers_info(self):
        return get_layers_info(self.layers)

    def get_wts_info(self, detailed=False):
        return get_wts_info((l.get_wts() for l in self.tr_layers), detailed)

    def get_training_params_info(self):
        return get_training_params_info(self.tr_prms)