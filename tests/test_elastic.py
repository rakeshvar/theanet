import ast
import sys
from PIL import Image as im
import matplotlib.pyplot as plt

import numpy as np
import theano.tensor as tt
from theano import shared, config, function
from theanet.layer import ElasticLayer

config.exception_verbosity = 'high'
config.optimizer = 'fast_compile'
np.set_printoptions(precision=2)
############################################## Helpers


def read_json_bz2(path2data, dtype=config.floatX):
    import bz2, json
    bz2_fp = bz2.BZ2File(path2data, 'r')
    data = np.array(json.loads(bz2_fp.read().decode('utf-8')), dtype=dtype)
    bz2_fp.close()
    return data


def share(data, dtype=config.floatX):
    return shared(np.asarray(data, dtype), borrow=True)


def pprint(slab):
    for r in slab:
        print(end='|')
        for val in r:
            if   val < 0.0:  print('-', end='')
            elif val == 0.:  print(' ', end='')
            elif val < .15:  print('·', end=''),
            elif val < .35:  print('░', end=''),
            elif val < .65:  print('▒', end=''),
            elif val < .85:  print('▓', end=''),
            elif val <= 1.:  print('█', end=''),
            else:            print('+', end='')
        print('|')


def stack(imgs3d):
    return np.rollaxis(imgs3d, 0, 1).reshape(img_sz*batch_sz, img_sz)

############################################## Arguments

if len(sys.argv) < 3:
    print("Usage:\n"
          "{} data.images.bz2 nnet_params.prms [begin_at]".format(sys.argv[0]))
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
deform_fn = function([x_sym], deform_layer.debugout)
print(deform_layer)

############################################## Perform deformation

if begin is None:
    begin = np.random.randint(corpus_sz-batch_sz, size=1)[0]
n_batches = 10

for index in range(begin, begin + n_batches * batch_sz, batch_sz):
    df_imgs, trans, *debug = deform_fn(data_x[index:index+batch_sz])
    for vec in debug:
        print(vec.flatten(), end=', ')

    if deform_layer.invert:
        df_imgs = 1 - df_imgs

    sidebyside = np.hstack((stack(data_x[index:index+batch_sz]),
                            stack(df_imgs)))

    out_img = data_file.replace('.bz2', str(index) + '.bmp')
    scaled = (255*(1-sidebyside)).astype('uint8')
    im.fromarray(scaled).save(out_img)

    plt.quiver(trans[1], trans[0])
    plt.savefig(out_img.replace('.bmp', '.png'))
    plt.clf()
    print("\nSaved ", out_img, out_img.replace('.bmp', '.png'))
