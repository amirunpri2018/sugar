import numpy
import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.linalg.ops import eigh, trace
from theano.tensor.nlinalg import alloc_diag
from theano.ifelse import ifelse
class sugar(object):
    def __init__(self, numpy_rng, x=None, y=None,
                 alpha=0.9,
                 lam=1.8 * 1e-3,
                 epsilon=1.8 * 1e-7,
                 sample_rate=0.1, n_visible=784, n_hidden=100,
                 W=None, bhid=None, bvis=None,
                 allX=None, allY=None, srng=None, P=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.epsilon = epsilon
        self.lam = lam
        self.sample_rate = sample_rate
        if not W:
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)),
                                      dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)
        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                                   dtype=theano.config.floatX),
                                 borrow=True)
        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden,
                                                   dtype=theano.config.floatX),
                                 name='b',
                                 borrow=True)
        if not srng:
            srng = RandomStreams(seed=234)
        if not P:
            initial_P = numpy.asarray(numpy_rng.uniform(
                        low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                        high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                        size=(n_visible, n_hidden)),
                                        dtype=theano.config.floatX)
            P = theano.shared(value=initial_P, name='P', borrow=True)
        self.srng = srng
        self.P = P
        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        if x == None:
            self.x = tensor.dmatrix(name='x')
        else:
            self.x = x
        if y == None:
            self.y = tensor.ivector(name='y')
        else:
            self.y = y
        if allX == None:
            self.allX = tensor.dmatrix(name='allX')
        else:
            self.allX = allX
        if allY == None:
            self.allY = tensor.ivector(name='allY')
        else:
            self.allY = allY
        self.params = [self.P, self.W, self.b, self.b_prime]
        self.dx, self.dy, self.length = self.down_sampleT(self.allX, self.allY, self.sample_rate)
    def get_hidden_values(self, input):
        return tensor.nnet.sigmoid(tensor.dot(input, self.W) + self.b)
    def get_reconstructed_input(self, hidden):
        return  tensor.nnet.sigmoid(tensor.dot(hidden, self.W_prime) + self.b_prime)
    def sgn_yT(self, y_i, y_j):
        return tensor.switch(tensor.eq(y_i,y_j), 1, -1)
    def down_sampleT(self, x, y, _sample_rate):
        length = tensor.cast(tensor.shape(y)[0] * _sample_rate, 'int32')
        id_max = tensor.cast(tensor.shape(y)[0] - 1, 'int32')
        def get_sub(i,x,y):
            idd = self.srng.random_integers(low = 0, high = id_max)
            return [x[idd], y[idd]]
        ([dx, dy], updates) = theano.scan(fn = get_sub,
                outputs_info=None,
                sequences=tensor.arange(length),
                non_sequences=[x,y])
        return dx, dy, length
    def get_J_AE(self):
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        J_AE = - tensor.sum(self.x * tensor.log(z) +
                             (1 - self.x) * tensor.log(1 - z),
                             axis=1)
        return tensor.mean(J_AE)
    def get_J_SH(self, x, y, P):
        length = tensor.cast(tensor.shape(y)[0], 'int32')
        Sv, updates = theano.scan(lambda ind, y, n: self.sgn_yT(y[ind // n], y[ind % n]),
                                      outputs_info=None,
                                      sequences=[tensor.arange(length ** 2)],
                                      non_sequences=[y,length]
                                     )
        S = tensor.reshape(Sv, (length, length))
        JP = tensor.dot(tensor.dot(tensor.dot( tensor.dot(P.T, x.T), S ), x), P)
        ret = trace(JP)
        ret = tensor.sum(ret) / tensor.sum(P ** 2) / self.length / self.length / 2.
        return -ret
    def get_cost_updates(self, contraction_level, learning_rate, switch, proj=True):
        self.J_AE = self.get_J_AE()
        self.J_SH = self.get_J_SH(self.dx, self.dy, self.P)
        self.J3 = self.lam * tensor.sum(tensor.abs_(self.W))
        self.J4 = self.epsilon * trace(tensor.dot(self.P - self.W, (self.P - self.W).T ))
        cost= self.alpha * self.J_AE + (1. - self.alpha) * self.J_SH + self.J3 + self.J4
        gparams = tensor.grad(cost, self.params)
        updates = []
        param=self.params[0]
        gparam=gparams[0]
        pp=param - learning_rate * gparam
        if proj:
            pp=self.update_ApprProj(pp, switch)
        updates.append((param, pp))
        for param, gparam in zip(self.params[1:], gparams[1:]):
            updates.append((param, param - learning_rate * gparam))
        return (cost, updates)
    def update_ApprProj(self, A, switch):
        S, U = eigh(tensor.dot(A, A.T), UPLO='L')
        length = tensor.cast(tensor.shape(S)[0], 'int32')
        def sqrt_inverse(v):
            return tensor.switch(tensor.le(v, 1e-8), 0., 1. / v)
        sqrtS, updates = theano.scan(lambda ind, S: sqrt_inverse(S[ind]),
                                      outputs_info=None,
                                      sequences=[tensor.arange(length)],
                                      non_sequences=[S]
                                     )
        diagS_inv=alloc_diag(sqrtS)
        AAA=tensor.dot( U.dot(diagS_inv).dot(U.T), A )
        return ifelse(tensor.eq(switch, 1), AAA, A)
