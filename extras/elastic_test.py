from __future__ import print_function
from time import time

import importlib
import numpy as np
import sys
from time import sleep
from neuralnet import DeformLayer
from deformer import Deformer, transform_flat_batch
from theano import shared, config, function
import theano.tensor as T
from PIL import Image as im
import matplotlib.pyplot as plt

def read_json_bz2(path2data):
    import bz2, json, contextlib

    with contextlib.closing(bz2.BZ2File(path2data, 'rb')) as fdata:
        return np.array(json.load(fdata))


def share(data, dtype=config.floatX):
    return shared(np.asarray(data, dtype), borrow=True)


prms_file_name = sys.argv[2]

if not prms_file_name.endswith('.py'):
    sys.exit()

params = importlib.import_module(prms_file_name[:-3])
layers = params.layers
tr_prms = params.training_params
allwts = None

print('{} : '.format(layers[0]))

batch_sz = tr_prms['BATCH_SZ']
img_sz = layers[0][1]['img_sz']
data_x = read_json_bz2(sys.argv[1])
data_x = data_x.reshape((data_x.shape[0], img_sz, img_sz))
print("X (samples, dimension) Size : {} {}KB\n".format(data_x.shape, data_x.nbytes // 1000,))

corpus_sz = data_x.shape[0]

imgs = share(data_x)
x = T.tensor3('x')
deform_layer = DeformLayer(x, **layers[0][1])
deform_fn = function([x], [deform_layer.output, deform_layer.trans])

at = np.random.randint(corpus_sz-batch_sz, size=1)[0]
tr_imgs, trans = deform_fn(data_x[at:at+batch_sz])
print (tr_imgs.shape)

def pprint(x):
    for r in x:
        for val in r:
            if val == 0:     print('#', end=''),
            elif val < .25:  print('@', end=''),
            elif val < .5:   print('O', end=''),
            elif val < .75:  print('o', end=''),
            elif val < 1.0:  print('.', end=''),
            else:            print(' ', end=''),
        print()
    sleep(1)

def stack(imgs3d):
    return np.rollaxis(imgs3d, 0, 1).reshape(img_sz*batch_sz, img_sz)

final = np.hstack((stack(data_x[at:at+batch_sz]), stack(tr_imgs)))
pprint(final)

plt.quiver(trans[0], trans[1])
plt.show()
im.fromarray((255*final).astype('uint8')).save('tmp/' + sys.argv[1][:-4] + str(at) + '.bmp')