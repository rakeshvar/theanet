#!/usr/bin/python
import multiprocessing as mp
import numpy as np
from scipy.ndimage.filters import gaussian_filter as gauss
from scipy.ndimage.interpolation import map_coordinates as mapcoords

def transform(img, scale, sigma, cval=0, ret_trans=False):
    '''
    Transforms a single 2D image
    '''
    trans = np.indices(img.shape) + \
                scale * np.random.uniform(-1, 1, (2,) + img.shape)
    for t in trans:
        gauss(t, sigma, output=t, mode='nearest', truncate=2)

    ret_img = mapcoords(img, trans, order=1, cval=cval, mode='constant')
    if ret_trans: return ret_img, trans
    else:         return ret_img

def transform_flat(img, shape, *args, **kwargs):
    return transform(img.reshape(shape), *args, **kwargs).flatten()

def transform_flat_batch(imgs, *args, **kwargs):
    return np.asarray([transform_flat(img, *args, **kwargs)
                            for img in imgs])

def transform_inplace(imgs, *args, **kwargs):
    imgs[:] = transform_flat_batch(imgs, *args, **kwargs)

class Deformer(object):
    """Deform a database of input images"""

    def __init__(self, data, batch_sz, img_shape, scale, sigma,
                    cval=0.0, ncpus=None):
        nBatches = data.shape[0] // batch_sz
        ncpus = mp.cpu_count() if (ncpus is None) else ncpus

        self.data = data
        self.batch_sz = batch_sz
        self.nBatches = nBatches
        self.img_shape = img_shape
        self.scale = scale
        self.sigma = sigma
        self.cval = cval
        self.ncpus = ncpus
        self.ndone = 0
        self.queue = mp.Queue(self.ncpus)

        self.processes = [mp.Process(target=self.build_batch, args=(i,))
                            for i in range(self.ncpus)]
        self.start_all()

    def build_batch(self, index):
        for b in range(index, self.nBatches, self.ncpus):
            transform_inplace(self.data[b * self.batch_sz:(b + 1) * self.batch_sz],
                              self.img_shape, self.scale, self.sigma, self.cval)
            self.queue.put(b)
        self.queue.put('done')

    def __str__(self):
        return ('Deformer: Input Shape {} batch_sz {} '
                'WH {} #Batches {} #cores {} '
                'Scale {} Sigma {} Background {} ').format(
                self.data.shape, self.batch_sz, self.img_shape,
                self.nBatches, self.ncpus,
                self.scale, self.sigma, self.cval)

    def start_all(self):
        for iproc in self.processes:
            iproc.daemon = True
            iproc.start()

    def __iter__(self):
        while self.ndone < self.ncpus:
            item = self.queue.get()
            if item == 'done':
                self.ndone += 1
            else:
                yield item


if __name__ == '__main__':
    from PIL import Image as im
    from scipy.ndimage import imread
    from guppy import hpy
    import sys
    import matplotlib.pyplot as plt

    try:
        file_name = sys.argv[1]
    except IndexError:
        print 'Usage {} imagefile [scale=64] [sigma=8]'.format(sys.argv[0])
        sys.exit(-1)

    try:
        scale = int(sys.argv[2])
    except IndexError:
        scale = 64
    try:
        sigma = int(sys.argv[3])
    except IndexError:
        sigma = 8

    def read_json_bz2(path2data):
        """

        :param path2data:
        :return: np.array
        """
        print "Loading ...",
        import bz2,json,contextlib
        with contextlib.closing(bz2.BZ2File(path2data, 'rb')) as f:
            return np.array(json.load(f))

    def pprint(x):
        for r in x:
            for val in r:
                if val == 0 :          print '#',
                elif val < .25 :       print '@',
                elif val < .5 :        print '0',
                elif val < .75 :       print 'o',
                elif val < 1.0 :       print '.',
                else:                  print ' ',
            print

    def test_one(img_name, img, scale, sigma, cval=255):
        img2, trans = transform(img, scale, sigma, ret_trans=True, cval=cval)
        tx, ty = trans

        print 'image : ', img_name, ' scale: ', scale, ' sigma: ', sigma
        print 'Scale', scale, 'sigma', sigma
        print "({:.2f}, {:.2f}) ({:.2f}, {:.2f}) ".format(
            tx.min(), tx.max(), ty.min(), ty.max())
        im.fromarray(np.vstack((img, img2))).save(img_name)

        plt.quiver(tx, ty)
        plt.imshow(np.hstack((img, img2)))

    ext = file_name.split('.')[-1]
    
    if ext != 'bz2':
        main_img = imread(file_name, mode="L")
        img_name = file_name.replace(
            '.' + ext, '_{}_{}.{}'.format(scale, sigma, ext))
        test_one(img_name, main_img, scale, sigma)
    
    else:
        import sharedmem
        main_imgs = read_json_bz2(file_name)
        main_imgs = main_imgs.astype('float')
        print main_imgs.dtype
        shrd_imgs = sharedmem.copy(main_imgs)
        wd = ht = int(main_imgs.shape[1] ** .5)

        batch_sz = 100
        h = hpy()
        deformer = Deformer(shrd_imgs, batch_sz, (ht, wd), scale, sigma, 1)
        print deformer
        for ibatch in deformer:
            print 'Processing Imgs : ', ibatch * batch_sz, '-', (ibatch+1 )* batch_sz,
            print shrd_imgs[ibatch*batch_sz].max(), shrd_imgs[ibatch*batch_sz].min()
            for i in range(batch_sz):
                iimg = ibatch * batch_sz + i
                img_name = 'tmp/' + file_name.replace(
                                        '.'+ext, '_{}.{}'.format(iimg, 'tif'))
                composite = np.vstack((main_imgs[iimg].reshape(ht, wd),
                                       shrd_imgs[iimg].reshape(ht, wd)))
                pprint(composite)
                im.fromarray((255*composite).astype('uint8')).save(img_name)
                break
        print h.heap()
