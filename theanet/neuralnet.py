import numpy as np
import theano
import theano.tensor as tt

from . import layer
from .layer import InputLayer, ElasticLayer, ColorLayer
from .layer import ConvLayer, PoolLayer, MeanLayer, DropOutLayer
from .layer import HiddenLayer, AuxConcatLayer
from .layer import SoftmaxLayer, CenteredOutLayer, SoftAuxLayer, HingeLayer

# theano.config.optimizer = 'fast_compile'
# theano.config.exception_verbosity = "high"

# ########################### Helper Functions #################################
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
            self.rand_gen = np.random.RandomState(training_params['SEED'])
        else:
            self.rand_gen = None

        self.tr_prms = training_params
        self.layers = layers
        self.allwts = allwts
        self.tr_layers = []
        self.te_layers = []
        self.batch_sz = training_params['BATCH_SZ']
        self.num_layers = 0

        # Symbolic variables for the data
        self.x = tt.tensor4('x')
        self.test_x = tt.tensor4('test_x')
        self.y = tt.ivector('y')

        # Input Layer
        input_layer_type = getattr(layer, layers[0][0])
        assert input_layer_type in (InputLayer, ElasticLayer, ColorLayer), \
            "First layer needs to be Input or Elastic or Color Layer"

        self.tr_layers.append(input_layer_type(self.x, rand_gen=self.rand_gen,
                                               **layers[0][1]))
        self.te_layers.append(self.tr_layers[0].TestVersion(self.test_x))
        self.num_layers += 1

        # Rest of the layers
        while self.num_layers < len(layers):
            self.append_next_layer()

        # Handle Auxiliary input
        for tr_layer, te_layer in zip(self.tr_layers, self.te_layers):
            if type(tr_layer) in (AuxConcatLayer, SoftAuxLayer):
                assert not hasattr(self, 'aux_inpt_tr'), "Multiple Aux Inputs"
                self.aux_inpt_tr = tr_layer.aux_inpt
                self.aux_inpt_te = te_layer.aux_inpt

        # Set Epoch and learning rate
        if 'CUR_EPOCH' not in training_params:
            training_params['CUR_EPOCH'] = 0
        self.cur_learn_rate = theano.shared(np.cast[theano.config.floatX](0.0))
        self.set_rate()

    def append_next_layer(self):
        layer_type, layer_args = self.layers[self.num_layers]
        prev_tr_layer = self.tr_layers[self.num_layers - 1]
        prev_te_layer = self.te_layers[self.num_layers - 1]
        wts = self.allwts[self.num_layers] if self.allwts else None

        tr_inpt = prev_tr_layer.output
        te_inpt = prev_te_layer.output
        curr_layer_type = getattr(layer, layer_type)

        if curr_layer_type in (ElasticLayer, ColorLayer,
                               ConvLayer, PoolLayer, MeanLayer):
            if type(prev_tr_layer) is DropOutLayer:
                use_tr_layer = self.tr_layers[self.num_layers - 2]
            else:
                use_tr_layer = prev_tr_layer
            num_prev_maps = use_tr_layer.num_maps
            prev_out_sz = use_tr_layer.out_sz

        if curr_layer_type in (ElasticLayer, ColorLayer):
            if "num_maps" in layer_args:
                del layer_args["num_maps"]
            if "img_sz" in layer_args:
                del layer_args["img_sz"]

            curr_layer = curr_layer_type(tr_inpt,
                                         num_maps=num_prev_maps,
                                         img_sz=prev_out_sz,
                                         rand_gen=self.rand_gen,
                                         **layer_args)

        elif curr_layer_type is ConvLayer:
            curr_layer = ConvLayer(tr_inpt,
                                   wts,
                                   self.rand_gen,
                                   self.batch_sz,
                                   num_prev_maps,
                                   prev_out_sz,
                                   **layer_args)

        elif curr_layer_type in (PoolLayer, MeanLayer):
            curr_layer = curr_layer_type(tr_inpt,
                                         num_maps=num_prev_maps,
                                         in_sz=prev_out_sz,
                                         **layer_args)

        elif curr_layer_type is DropOutLayer:
            curr_layer = DropOutLayer(tr_inpt,
                                      self.rand_gen,
                                      prev_tr_layer.n_out,
                                      **layer_args)

        elif curr_layer_type in (AuxConcatLayer, HiddenLayer,
                                 SoftmaxLayer, SoftAuxLayer, HingeLayer):
            te_inpt = te_inpt.flatten(2)
            curr_layer = curr_layer_type(tr_inpt.flatten(2),
                                         wts,
                                         self.rand_gen,
                                         prev_tr_layer.n_out,
                                         **layer_args)

        elif curr_layer_type is CenteredOutLayer:
            """
            CenteredOutLayer
                Can be LOGIT or RBF
                Needs Wts for hidden layer
                (can be generated randomly as nPrevLayerUnits x nFeatures)
                Needs CENTERS as a matrix or as a tuple
                (nClasses x nFeatures) to convert from Features to Class probs.
            """
            if wts:
                centers = wts[3]
                wts = wts[:2]
            else:
                centers = None

            te_inpt = te_inpt.flatten(2)
            curr_layer = CenteredOutLayer(tr_inpt.flatten(2),
                                          wts, centers,
                                          self.rand_gen,
                                          prev_tr_layer.n_out,
                                          **layer_args)
        else:
            raise NotImplementedError("Unknown Layer Type" + layer_type)

        self.tr_layers.append(curr_layer)
        self.te_layers.append(curr_layer.TestVersion(te_inpt))
        self.num_layers += 1

    def get_trin_model(self, x_data, y_data, aux_data=None,
                       take_index_list=False):
        # cost, training params, gradients, updates to those params
        print('Compiling training function...')

        cost = self.tr_layers[-1].cost(self.y)
        for lyr in self.tr_layers:
            cost += lyr.get_wtcost()

        updates = []
        for lyr in self.tr_layers:
            updates += lyr.get_updates(cost, self.cur_learn_rate)

        if hasattr(self, 'aux_inpt_tr'):
            assert aux_data is not None, "Auxillary data not supplied"

        if not take_index_list:
            indx = tt.lscalar('training batch index')
            bsz = self.tr_prms['BATCH_SZ']
            givens = {
                self.x: x_data[indx * bsz:(indx + 1) * bsz],
                self.y: y_data[indx * bsz:(indx + 1) * bsz], }
            if hasattr(self, 'aux_inpt_tr'):
                givens[self.aux_inpt_tr] = aux_data[indx * bsz:(indx + 1) * bsz]

        else:
            indx = tt.ivector('training batch indices')
            givens = {
                self.x: x_data[indx],
                self.y: y_data[indx], }
            if hasattr(self, 'aux_inpt_tr'):
                givens[self.aux_inpt_tr] = aux_data[indx]

        return theano.function([indx],
                               [cost,
                                self.tr_layers[-1].features,
                                self.tr_layers[-1].logprob],
                               updates=updates,
                               givens=givens)

    def reset_accumulated_gradients(self):
        if not hasattr(self, "accumulated_gradients_resetter"):
            accumulated_updates = []
            for lyr in self.tr_layers:
                for au in lyr.accumulated_updates:
                    #au.set_value(0*au.get_value())
                    accumulated_updates.append((au, 0*au))

            print("Compiling accumulated_gradients_resetter")
            self.accumulated_gradients_resetter = theano.function([],
                updates=accumulated_updates)

        self.accumulated_gradients_resetter()

    def get_test_model(self, x_data, y_data, aux_data=None, preds_feats=False):
        print('Compiling testing function... ')
        idx = tt.lscalar('test batch index')
        bth_sz = self.tr_prms['BATCH_SZ']

        givens = {
            self.test_x: x_data[idx * bth_sz:(idx + 1) * bth_sz],
            self.y: y_data[idx * bth_sz:(idx + 1) * bth_sz]}

        if hasattr(self, 'aux_inpt_te'):
            assert aux_data is not None, "Auxillary data not supplied"
            givens[self.aux_inpt_te] = \
                aux_data[idx * bth_sz:(idx + 1) * bth_sz]

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