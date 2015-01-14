from __future__ import print_function

import ast
import sys
from layer import ElasticLayer

import numpy as np
import theano.tensor as tt
from theano import shared, config, function
from PIL import Image as im
import matplotlib.pyplot as plt

def read_json_bz2(path2data):
    import bz2, json, contextlib

    with contextlib.closing(bz2.BZ2File(path2data, 'rb')) as fdata:
        return np.array(json.load(fdata), dtype=config.floatX)


def share(data, dtype=config.floatX):
    return shared(np.asarray(data, dtype), borrow=True)


prms_file_name = sys.argv[2]

with open(prms_file_name, 'r') as p_fp:
    params = ast.literal_eval(p_fp.read())


layers = params['layer']
tr_prms = params['training_params']
batch_sz = tr_prms['BATCH_SZ']

print('Loading Data')
data_x = read_json_bz2(sys.argv[1])
corpus_sz, layers[0][1]['img_sz'], img_sz = data_x.shape
print(layers[0])
print("X (samples, dimension) Size : {} {}KB\n".format(
    data_x.shape, data_x.nbytes // 1000,))
print(data_x[0,:16,:16])

imgs = share(data_x)
x = tt.tensor3('x')
deform_layer = ElasticLayer(x, **layers[0][1])
deform_fn = function([x], [deform_layer.output, deform_layer.trans])

at = np.random.randint(corpus_sz-batch_sz, size=1)[0]
tr_imgs, trans = deform_fn(data_x[at:at+batch_sz])
print (tr_imgs.shape)

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

tr_imgs = np.round(tr_imgs)
final = 1-np.hstack((stack(data_x[at:at+batch_sz]), stack(tr_imgs)))
pprint(final[:96])

plt.quiver(trans[0], trans[1])
plt.show()
im.fromarray((255*final).astype('uint8')).save(sys.argv[1][:-4] + str(at) + '.bmp')

for i in range(32):
    for j in range(12,20):
        print('({:6.2f} {:6.2f})'.format(trans[0,i,j]-i,
                                         trans[1,i,j]-j,), end=' ')
    print()