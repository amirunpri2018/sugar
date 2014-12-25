import sys, cPickle, gzip, os
import numpy
from theano import tensor, shared, config
def pickle(file, data):
    fo = gzip.open(file, 'wb')
    cPickle.dump(data, fo)
    fo.close()
    return dict
def savePm(output="p1.pkl.gz", W=None, bvis=None, bhid=None):
    data = {'W': W, 'bvis': bvis, 'bhid': bhid}
    pickle(output, data)
    return
def getNumZero(W=None, delta=1e-3):
    sum=0
    for i in W.flatten() :
        if abs(i) < delta :
            sum+=1
    return 100.*sum/W.shape[0]/W.shape[1]
def unpickle(file):
    fo = gzip.open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
def get_WW(dataset='WW.pkl.gz'):
    data = unpickle(dataset)
    W=data['W']
    b=data['bhid']
    return W, b
def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist_rotation_back_image_new.pkl.gz':
        import urllib
        origin = 'http://kdd2014.noahlab.com.hk/sugar/mnist_rotation_back_image_new.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)
    print '... loading data'
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = shared(numpy.asarray(data_x, dtype=config.floatX),
                          borrow=borrow)
        shared_y = shared(numpy.asarray(data_y,dtype=config.floatX),
                          borrow=borrow)
        return shared_x, tensor.cast(shared_y, 'int32')
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
