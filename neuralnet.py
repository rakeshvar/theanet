from __future__ import print_function
import numpy as np
import theano
import theano.tensor        as T
from   theano.tensor.signal import downsample
from   theano.tensor.nnet import conv

# ##############################################################################
#   A bunch of Helper Functions
###############################################################################


def is_shared_var(x):
    """
    Checks if x is SharedVar.
    Could be a CUDA SharedVar or just a normal Theano SharedVar
    """
    return 'SharedVariable' in str(type(x))


def init_wts(wts, rand_gen, size_w, size_b, fan_in, fan_out, actvn):
    """
    Initialize the weights for any layer
    :param wts:      Could be None, numpy array or another shared variable
    :param rand_gen: Needed if wts is None to generate new weights
    :param size_w:     - do -
    :param size_b:     - do -
    :param fan_in:    - do -
    :param fan_out:   - do -
    :param actvn:     - do -
    :return:
    """
    if wts is None:
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        w_values = np.asarray(
            rand_gen.uniform(low=-w_bound, high=w_bound, size=size_w),
            dtype=theano.config.floatX)
        b_values = np.zeros(size_b, dtype=theano.config.floatX)

        if actvn == 'sigmoid': w_values *= 4
        if actvn in ('relu', 'softplus'):  b_values += 1

    elif type(wts[0]) is np.ndarray:
        w_values, b_values = wts

    if wts and is_shared_var(wts[0]):  # Usually true when getting TestVersion
        W, b = wts
    else:
        W = theano.shared(w_values, name='W', borrow=True)
        b = theano.shared(b_values, name='b', borrow=True)

    return W, b


def borrow(sharedvar, boro=True):
    """
    Gets the numpy ndarray underlying a sharedVariable
    """
    return sharedvar.get_value(borrow=boro)


###############################################################################
#   A bunch of Activations
###############################################################################

class Activation:
    """
    Defines a bunch of activations as callable classes.
    Useful for printing and specifying activations as strings.
    """
    def __init__(self, fn, name):
        self.fn = fn
        self.name = name

    def __call__(self, *args):  return self.fn(*args)

    def __str__(self):          return self.name


scaled_tanh = Activation(lambda x: 1.7 * T.tanh(2 * x / 3), 'scaled_tanh')
tanh = Activation(lambda x: T.tanh(x), 'tanh')
relu = Activation(lambda x: T.maximum(0, x), 'relu')

activation_list = ( scaled_tanh, relu,
                    T.nnet.sigmoid, tanh,
                    T.nnet.softplus, T.nnet.softmax)


def activation_by_name(name):
    """
    Get an activation function or callabe-class from its name
    :param name: string
    :return: Callable Activation
    """
    for act in activation_list:
        if name == str(act):
            return act
    else:
        raise NotImplementedError


###############################################################################
#   A bunch of Layers
###############################################################################

class Layer(object):
    """
    Base class for Layer
    """
    def __str__(self):
        return self.representation

    def get_wts(self):
        return ()


############################### Input Layer  ##################################

def perturb(inpt, batch_sz, img_sz, new_sz, pert_x, pert_y):
    return inpt.reshape((batch_sz, img_sz, img_sz)) \
        [:, pert_x:pert_x + new_sz, pert_y:pert_y + new_sz] \
        .reshape((batch_sz, new_sz * new_sz))


class InputLayer(Layer):
    def __init__(self, inpt,
                 rand_gen=None,
                 pflip=0,
                 max_perturb=0,
                 img_sz=0, batch_sz=0,
                 num_maps=1):
        self.params = []

        if pflip or max_perturb:
            if rand_gen:
                shrd_rnd_strm = T.shared_randomstreams.RandomStreams(rand_gen.randint(1e7))
            else:
                shrd_rnd_strm = T.shared_randomstreams.RandomStreams()

        if pflip == 0:
            self.noised_inpt = inpt
        else:
            mask = shrd_rnd_strm.binomial(n=1, p=pflip, size=inpt.shape)
            self.noised_inpt = T.cast(T.cast(inpt, 'int32') ^ mask, theano.config.floatX)

        out_sz = img_sz - max_perturb
        if max_perturb == 0:
            self.output = self.noised_inpt
        else:
            assert img_sz and batch_sz
            pert_x = shrd_rnd_strm.random_integers(low=0, high=max_perturb)
            pert_y = shrd_rnd_strm.random_integers(low=0, high=max_perturb)
            self.output = perturb(self.noised_inpt, batch_sz, img_sz, out_sz,
                                  pert_x, pert_y)

        self.inpt = inpt
        self.pflip = pflip
        self.max_perturb = max_perturb
        self.batch_sz = batch_sz
        self.img_sz = img_sz
        self.out_sz = out_sz
        self.num_maps = num_maps
        self.representation = \
            'Input\tMaps:{:2d}\tSize:{:2d}\tFlip%:{:}\tPerturbation:{}\tOutput:{:2d}'. \
                format(num_maps, img_sz, pflip, max_perturb, out_sz)

    def TestVersion(self, inpt):
        perturbed_inpt = perturb(inpt, self.batch_sz, self.img_sz,
                                 self.img_sz - self.max_perturb,
                                 self.max_perturb // 2, self.max_perturb // 2)
        test_version = InputLayer(perturbed_inpt)
        test_version.out_sz = self.out_sz
        return test_version


############################### ConvPool Layer ################################

class ConvPoolLayer(Layer):
    def __init__(self, inpt, wts, rand_gen,
                 batch_sz, num_prev_maps, in_sz,
                 num_maps, filter_sz, stride, pool_sz,
                 actvn='tanh'):
        assert (wts != None or rand_gen != None)
        image_shape = (batch_sz, num_prev_maps, in_sz, in_sz)
        filter_shape = (num_maps, num_prev_maps, filter_sz, filter_sz)

        # Assign Weights
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_sz)
        self.W, self.b = init_wts(wts, rand_gen,
                                  filter_shape, (filter_shape[0], ),
                                  fan_in, fan_out, actvn)

        # Add ConvPool Operation to the graph
        conv_out = conv.conv2d(inpt, self.W, image_shape, filter_shape,
                               subsample=(stride, stride))
        conv_pool = downsample.max_pool_2d(conv_out, (pool_sz, pool_sz),
                                           ignore_border=True)
        self.output = activation_by_name(actvn)(conv_pool + self.b.dimshuffle('x', 0, 'x', 'x'))

        # Calculate output shape
        self.out_sz = (in_sz - filter_sz + 1) / (stride * pool_sz)

        # Store Parameters
        self.params = [self.W, self.b]
        self.inpt = inpt
        self.representation = \
            'ConvPool\tMaps:{:2d}\tFilter:{}\tStride:{}\tPool:{}\tOutput:{:2d}\tAct:{}'. \
                format(num_maps, filter_sz, stride, pool_sz, self.out_sz, actvn)
        self.args = (batch_sz, num_prev_maps, in_sz,
                     num_maps, filter_sz, stride, pool_sz, actvn)
        self.num_maps = num_maps

    def TestVersion(self, inpt):
        return ConvPoolLayer(inpt, (self.W, self.b), None, *self.args)

    def get_wts(self):
        return (borrow(self.W), borrow(self.b))


############################### Hidden Layer  ##################################

def drop_output(output, pdrop, rand_gen):
    if rand_gen == None:
        shrd_rnd_strm = T.shared_randomstreams.RandomStreams()
    else:
        shrd_rnd_strm = T.shared_randomstreams.RandomStreams(rand_gen.randint(1e7))

    mask = shrd_rnd_strm.binomial(n=1, p=1 - pdrop, size=output.shape)
    dropped_output = output * T.cast(mask, theano.config.floatX)
    return dropped_output


class HiddenLayer(Layer):
    def __init__(self, inpt, wts, rand_gen=None, n_in=None, n_out=None, pdrop=0,
                 actvn='softplus'):
        assert wts != None or rand_gen != None

        try:
            fan_in_out = n_in + n_out
        except TypeError:
            fan_in_out = None

        self.W, self.b = init_wts(wts, rand_gen, (n_in, n_out), (n_out,),
                                  fan_in_out, fan_in_out, actvn)
        n_in, n_out = borrow(self.W).shape

        self.output = activation_by_name(actvn)((T.dot(inpt, self.W) + self.b))
        if pdrop:
            self.output = drop_output(self.output, pdrop, rand_gen)

        self.inpt = inpt
        self.params = [self.W, self.b]
        self.n_in, self.n_out = n_in, n_out
        self.actvn = actvn
        self.pdrop = pdrop
        self.representation = "Hidden\tIn:{:3d}\tOut:{:3d}\tAct:{}\tDrop%:{}". \
            format(n_in, n_out, actvn, pdrop)

    def TestVersion(self, inpt):
        test_version = HiddenLayer(inpt, (self.W, self.b),
                                   pdrop=0,
                                   actvn=self.actvn)
        test_version.output *= 1 - self.pdrop
        return test_version

    def get_wts(self):
        return (borrow(self.W), borrow(self.b))


############################### Output Layer  ##################################

class OutputLayer(object):
    def neg_log_likli(self, y):
        return -T.mean(self.logprob[T.arange(y.shape[0]), y])

    def get_predictions(self, y):
        return (self.features, self.y_preds, y)

    def errors(self, y):
        sym_err_rate = T.mean(T.neq(self.y_preds, y))

        if self.kind == 'LOGIT':  # Bit error rate
            second_stat = T.mean(self.bitprob[T.arange(y.shape[0]), y] < .5)
        else:  # Likelihood of MLE
            second_stat = T.mean(self.probs[T.arange(y.shape[0]), y])

        return (sym_err_rate, second_stat, self.y_preds, y)


class SoftmaxLayer(HiddenLayer, OutputLayer):
    def __init__(self, inpt, wts, rand_gen=None, n_in=None, n_out=None):
        HiddenLayer.__init__(self, inpt, wts, rand_gen, n_in, n_out,
                             actvn='Softmax',
                             pdrop=0)
        self.y_preds = T.argmax(self.output, axis=1)
        self.probs = self.output
        self.logprob = T.log(self.probs)
        self.features = self.logprob
        self.kind = 'SOFTMAX'
        self.representation = 'Softmax\tIn:{:3d}\tOut:{:3d}'.format(self.n_in, self.n_out)

    def TestVersion(self, inpt):
        return SoftmaxLayer(inpt, (self.W, self.b))


activs = {'LOGIT': 'sigmoid', 'RBF': 'scaled_tanh'}


class CenteredOutLayer(HiddenLayer, OutputLayer):
    def __init__(self, inpt, wts, centers, rand_gen=None,
                 n_in=None, n_features=None, n_classes=None,
                 kind='LOGIT', learn_centers=False, junk_dist=np.inf):
        # wts (n_in x n_features)
        # centers (n_classesx n_features)

        assert kind in activs
        assert n_in or wts
        assert n_features or wts or centers
        assert n_classes or centers
        assert kind == 'RBF' or not learn_centers

        HiddenLayer.__init__(self, inpt, wts, rand_gen, n_in, n_out=n_features,
                             actvn=activs[kind], pdrop=0)

        # Initialize centers
        if centers is None:
            if kind == 'LOGIT':
                centers_vals = rand_gen.binomial(n=1, p=.5, size=(n_classes, n_features))
            elif kind == 'RBF':
                centers_vals = rand_gen.uniform(low=0, high=1, size=(n_classes, n_features))
            centers = np.asarray(centers_vals, dtype=theano.config.floatX)

        if is_shared_var(centers):
            self.centers = centers
        else:
            self.centers = theano.shared(centers, name='centers', borrow=True)

        if learn_centers:
            self.params.append(self.centers)

        # 
        if not n_in or not n_features:
            n_in, n_features = borrow(self.W).shape
        if not n_features or not n_classes:
            n_classes, n_features = borrow(self.centers).shape

        # c = centers; v = output of hidden layer = calculated features
        self.features = self.output  # Refers to the output of HiddenLayer
        c = self.centers.dimshuffle('x', 0, 1)
        v = self.features.dimshuffle(0, 'x', 1)
        self.kind = kind
        self.junk_dist = junk_dist

        if kind == 'LOGIT':
            # BATCH_SZ x nClasses x nFeatures >> BATCH_SZ x nClasses >> BATCH_SZ
            EPSILON = .001
            v = v * (1 - 2 * EPSILON) + EPSILON
            self.bitprob = c * v + (1 - c) * (1 - v)
            self.logprob = T.sum(T.log(self.bitprob), axis=2)
            # if imp == None \
            # else T.tensordot(T.log(self.bitprob), imp, axes=([2, 0]))
            self.y_preds = T.argmax(self.logprob, axis=1)
        elif kind == 'RBF':
            dists = T.sum((v - c) ** 2, axis=2)  # BATCH_SZ x nClasses
            junk_col = junk_dist + T.zeros_like(dists[:, 1]).dimshuffle(0, 'x')
            self.dists = T.concatenate([dists, junk_col], axis=1)
            self.probs = T.nnet.softmax(-self.dists)  # BATCH_SZ x nClasses+1
            self.logprob = T.log(self.probs)
            self.y_preds = T.argmax(self.probs, axis=1)

        self.representation = \
            'CenteredOut\tKind:{}\tIn:{:3d}\tHidden:{:3d}\tOut:{:3d}\tlearn_centers:{}\tjunk_dist:{}' \
                .format(kind, n_in, n_features, n_classes, learn_centers, junk_dist)

    def TestVersion(self, inpt):
        return CenteredOutLayer(inpt, (self.W, self.b), self.centers,
                                kind=self.kind, junk_dist=self.junk_dist)

    def get_wts(self):
        return (borrow(self.W), borrow(self.b), borrow(self.centers))


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
        index = T.lscalar()
        x = T.matrix('x')
        test_x = T.matrix('test_x')
        y = T.ivector('y')

        tr_layers = []
        te_layers = []

        # Input Layer
        assert layers[0][0] == 'InputLayer'
        batch_sz = layers[0][1]['batch_sz']

        ilayer = 0
        tr_layers.append(InputLayer(x, **layers[0][1]))
        te_layers.append(tr_layers[0].TestVersion(test_x))
        ilayer += 1

        # ConvPool Layers
        while (layers[ilayer][0] == 'ConvPoolLayer'):
            prev_tr_layer = tr_layers[ilayer - 1]
            prev_te_layer = te_layers[ilayer - 1]
            tr_inpt = prev_tr_layer.output
            te_inpt = prev_te_layer.output

            if isinstance(prev_tr_layer, InputLayer):
                img_sz = prev_tr_layer.out_sz
                tr_inpt = tr_inpt.reshape((batch_sz, 1, img_sz, img_sz))
                te_inpt = te_inpt.reshape((batch_sz, 1, img_sz, img_sz))

            wts = allwts[ilayer] if allwts else None

            curr_layer = ConvPoolLayer(tr_inpt, wts, rand_gen, batch_sz,
                                       prev_tr_layer.num_maps, prev_tr_layer.out_sz,
                                       **layers[ilayer][1])

            tr_layers.append(curr_layer)
            te_layers.append(curr_layer.TestVersion(te_inpt))
            ilayer += 1

        # Hidden Layers
        while (layers[ilayer][0] == 'HiddenLayer'):
            prev_tr_layer = tr_layers[ilayer - 1]
            prev_te_layer = te_layers[ilayer - 1]
            tr_inpt = prev_tr_layer.output
            te_inpt = prev_te_layer.output

            if isinstance(prev_tr_layer, ConvPoolLayer):
                tr_inpt = tr_inpt.flatten(2)
                te_inpt = te_inpt.flatten(2)

            if isinstance(prev_tr_layer, HiddenLayer):
                n_in = prev_tr_layer.n_out
            else:
                n_in = prev_tr_layer.num_maps * prev_tr_layer.out_sz ** 2

            wts = allwts[ilayer] if allwts else None

            curr_layer = HiddenLayer(tr_inpt, wts, rand_gen, n_in,
                                     **layers[ilayer][1])
            tr_layers.append(curr_layer)
            te_layers.append(curr_layer.TestVersion(te_inpt))
            ilayer += 1

        # Output layer
        #   |- CenteredOutLayer 
        #       |- Can be LOGIT or RBF
        #       |- Needs Wts for hidden layer (can be generated randomly as nPrevLayerUnits x nFeatures)
        #       |- Needs CENTERS as a matrix or as a tuple (nClassesxnFeatures) to convert from Features to Class probs
        #   |- SoftmaxLayer
        #       |- Needs Wts for hidden layer (can be generated randomly as nPrevLayerUnits x nClasses)
        #       |- Needs CENTERS as nClasses
        assert layers[ilayer][0] in ('SoftmaxLayer', 'CenteredOutLayer')

        wts = allwts[ilayer] if allwts else None
        prev_tr_layer = tr_layers[ilayer - 1]
        prev_te_layer = te_layers[ilayer - 1]
        assert isinstance(prev_tr_layer, HiddenLayer)

        if layers[ilayer][0] == 'SoftmaxLayer':
            curr_layer = SoftmaxLayer(prev_tr_layer.output, wts, rand_gen,
                                      prev_tr_layer.n_out, layers[ilayer][1]['n_out'])

        elif layers[ilayer][0] == 'CenteredOutLayer':
            try:
                centers = layers[ilayer][1].pop('centers')
            except KeyError:
                centers = None

            if allwts and len(allwts) == 3:
                wts = allwts[:2]
                centers = allwts[3]

            curr_layer = CenteredOutLayer(prev_tr_layer.output, wts, centers,
                                          rand_gen, prev_tr_layer.n_out, **layers[ilayer][1])

        tr_layers.append(curr_layer)
        te_layers.append(curr_layer.TestVersion(prev_te_layer.output))
        ilayer += 1

        # Store variables that need to be persistent
        self.tr_layers = tr_layers
        self.te_layers = te_layers
        self.num_layers = ilayer
        self.training_params = training_params
        self.layers = layers
        self.x = x
        self.y = y
        self.test_x = test_x
        self.batch_sz = batch_sz

        # Set Epoch and learning rate
        if 'CUR_EPOCH' not in training_params:    training_params['CUR_EPOCH'] = 0
        self.cur_learn_rate = theano.shared(np.cast[theano.config.floatX](0.0))
        self.set_rate()

    def get_trin_model(self):
        # cost, training params, gradients, updates to those params
        print('Compiling training function...')

        trn_prms = [prm for lyr in self.tr_layers for prm in lyr.params]
        cost = self.tr_layers[-1].neg_log_likli(self.y) + \
               self.training_params['LAMBDA1'] * sum(abs(t).sum() for t in trn_prms) + \
               self.training_params['LAMBDA2'] * sum((t ** 2).sum() for t in trn_prms)

        updates = []
        for param in trn_prms:
            param_update = theano.shared(borrow(param) * 0., broadcastable=param.broadcastable)
            updates.append((param_update,
                            self.training_params['MOMENTUM'] * param_update + (
                                1. - self.training_params['MOMENTUM']) * T.grad(cost, param)))
            stepped_param = param - self.cur_learn_rate * param_update

            if borrow(param).ndim == 2:
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, self.training_params['MAXNORM'])
                scale = (1e-7 + desired_norms) / (1e-7 + col_norms)
                updates.append((param, stepped_param * scale))
            else:
                updates.append((param, stepped_param))

        return theano.function(
            [self.x, self.y],
            [cost, self.tr_layers[-2].inpt,
             self.tr_layers[-1].inpt,
             self.tr_layers[-1].features,
             self.tr_layers[-1].logprob,
            ],
            updates=updates)

    def get_test_model(self):
        print('Compiling testing function... ')
        return theano.function([self.test_x, self.y], self.te_layers[-1].errors(self.y))

    def get_full_test_model(self, test_set_x, test_set_y):
        print('Compiling full testing function... ')
        return theano.function([self.test_x, self.y], self.te_layers[-1].get_predictions(self.y))

    def get_init_params(self):
        return {"layers": self.layers,
                "training_params": self.training_params,
                "allwts": [l.get_wts() for l in self.tr_layers]}

    def set_rate(self):
        self.cur_learn_rate.set_value(self.training_params['INIT_LEARNING_RATE'] /
                                      (1 + self.training_params['CUR_EPOCH'] / self.training_params[
                                          'EPOCHS_TO_HALF_RATE']))

    def inc_epoch_set_rate(self):
        self.training_params['CUR_EPOCH'] += 1
        self.set_rate()

    def get_epoch(self):
        return self.training_params['CUR_EPOCH']

    def __str__(self):
        prmstr = '; '.join([', '.join([str(prm) for prm in lyr.params]) for lyr in self.tr_layers])
        return \
            '\nTrain Layers\n\t' + \
            '\n\t'.join([str(l) for l in self.tr_layers]) + \
            '\nTest Layers\n\t' + \
            '\n\t'.join([str(l) for l in self.te_layers]) + \
            '\nParams ' + prmstr
