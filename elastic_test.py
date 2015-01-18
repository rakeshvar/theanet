from __future__ import print_function

import ast
import sys
from PIL import Image as im
import matplotlib.pyplot as plt

import numpy as np
import theano.tensor as tt
from theano import shared, config, function
from layer import ElasticLayer

############################################## Helpers


def read_json_bz2(path2data):
    import bz2, json, contextlib
    with contextlib.closing(bz2.BZ2File(path2data, 'rb')) as fdata:
        return np.array(json.load(fdata), dtype=config.floatX)


def share(data, dtype=config.floatX):
    return shared(np.asarray(data, dtype), borrow=True)


def pprint(x):
    for r in x:
        for val in r:
            if val < 0:      print('+', end='')
            elif val == 0:   print('#', end='')
            elif val < .25:  print('@', end='')
            elif val < .5:   print('O', end='')
            elif val < .75:  print('o', end='')
            elif val < 1.0:  print('.', end='')
            elif val == 1.:  print(' ', end='')
            else:            print('-', end='')
        print()


def stack(imgs3d):
    return np.rollaxis(imgs3d, 0, 1).reshape(img_sz*batch_sz, img_sz)

############################################## Arguments

if len(sys.argv) < 3:
    print("Usage:\n"
          "{} data.images.bz2 nnet_params.prms".format(sys.argv[0]))
    sys.exit(-1)

data_file = sys.argv[1]
prms_file_name = sys.argv[2]
try:
    begin = int(sys.argv[3])
except IndexError:
    begin = None

with open(prms_file_name, 'r') as p_fp:
    params = ast.literal_eval(p_fp.read())
    layers = params['layers']
    tr_prms = params['training_params']
    batch_sz = 10 #tr_prms['BATCH_SZ']

############################################## Load

print('Loading Data')
data_x = read_json_bz2(data_file)
corpus_sz, layers[0][1]['img_sz'], img_sz = data_x.shape

print(layers[0])
print("Images (samples, dimension) Size : {} {}KB\n"
      "Min: {} Max: {}, Mean: {}\n"
      "".format(data_x.shape, data_x.nbytes // 1000,
                data_x.min(), data_x.max(), data_x.mean()))

############################################## Init Layer

imgs = share(data_x)
x_sym = tt.tensor3('x')
deform_layer = ElasticLayer(x_sym, **layers[0][1])
deform_fn = function([x_sym], [deform_layer.output, deform_layer.trans])
print(deform_layer)

############################################## Perform deformation

if begin is None:
    begin = np.random.randint(corpus_sz-batch_sz, size=1)[0]
n_batches = 3

for index in range(begin, begin + n_batches * batch_sz, batch_sz):
    df_imgs, trans = deform_fn(data_x[index:index+batch_sz])
    if deform_layer.invert:
        df_imgs = 1 - df_imgs
    sidebyside = np.hstack((stack(data_x[index:index+batch_sz]),
                            stack(df_imgs)))

    out_img = data_file.replace('.bz2', str(index) + '.bmp')
    scaled = (255*(1-sidebyside)).astype('uint8')
    im.fromarray(scaled).save(out_img)

    plt.quiver(trans[0], trans[1])
    plt.savefig(out_img.replace('.bmp', '.png'))
