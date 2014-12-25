import os, sys, time
import numpy
from theano import tensor, function, Param
from theano.tensor.shared_randomstreams import RandomStreams
from layers import HiddenLayer, LogisticRegression
from SUGAR import sugar
from myio  import load_data, getNumZero

class deep_sugar(object):
    def __init__(self, numpy_rng, theano_rng=None, y=None, 
                 alpha=0.9, sample_rate=0.1, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10,
                 corruption_levels=[0.1, 0.1],
                 allX=None,allY=None,srng=None):
        self.sigmoid_layers = []
        self.sugar_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.allXs = []
        if y == None:
            self.y = tensor.ivector(name='y')
        else:
            self.y = y
        assert self.n_layers > 0
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.x = tensor.matrix('x')  
        self.x = tensor.matrix('x')  
        self.y = tensor.ivector('y')  
        self.y = tensor.ivector('y')  
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output
            if i == 0:
                self.allXs.append(allX)
            else:
                self.allXs.append(tensor.dot(self.allXs[i-1], self.sigmoid_layers[-1].W) + self.sigmoid_layers[-1].b)
            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=tensor.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)
            sugar_layer = sugar(numpy_rng=numpy_rng,
                                alpha=alpha,
                                sample_rate=sample_rate,
                                x=layer_input,
                                y=self.y,
                                n_visible=input_size,
                                n_hidden=hidden_layers_sizes[i],
                                W=sigmoid_layer.W,
                                bhid=sigmoid_layer.b,
                                allX=self.allXs[i],
                                allY=allY,
                                srng=srng)
            self.sugar_layers.append(sugar_layer)
        self.logLayer = LogisticRegression(
                         input=self.sigmoid_layers[-1].output,
                         n_in=hidden_layers_sizes[-1], n_out=n_outs)
        self.params.extend(self.logLayer.params)
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)
        
    def pretraining_functions(self, train_set_x, train_set_y, batch_size):
        index = tensor.lscalar('index')  
        index = tensor.lscalar('index')  
        corruption_level = tensor.scalar('corruption')  
        corruption_level = tensor.scalar('corruption')  
        learning_rate = tensor.scalar('lr')  
        learning_rate = tensor.scalar('lr')  
        switch = tensor.iscalar('switch')
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size
        pretrain_fns = []
        for sugar in self.sugar_layers:
            cost, updates = sugar.get_cost_updates(corruption_level,
                                                learning_rate,
                                                switch)
            fn = function(inputs=[index,
                                         Param(corruption_level, default=0.2),
                                         Param(learning_rate, default=0.1),
                                         Param(switch, default=1)],
                                 outputs=[cost],
                                 updates=updates,
                                 givens={self.x: train_set_x[batch_begin:batch_end],
                                         self.y: train_set_y[batch_begin:batch_end]}, on_unused_input='ignore')
            pretrain_fns.append(fn)
        return pretrain_fns
        
    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y)   = datasets[2]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size
        index = tensor.lscalar('index')  
        gparams = tensor.grad(self.finetune_cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))
        train_fn = function(inputs=[index],
              outputs=self.finetune_cost,
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})
        test_score_i = function([index], self.errors,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})
        valid_score_i = function([index], self.errors,
              givens={
                 self.x: valid_set_x[index * batch_size:
                                     (index + 1) * batch_size],
                 self.y: valid_set_y[index * batch_size:
                                     (index + 1) * batch_size]})
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]
        return train_fn, valid_score, test_score
        
def train(pretrain_lr=0.003,
          pretraining_epochs=50, 
          pretrain_batch_size=10,
          finetune_lr=0.03, 
          training_epochs=300, 
          batch_size=1,
          dataset='../data/mnist_rotation_back_image_new.pkl.gz',
          hidden_layers_sizes=[500, 500, 1000],
          alpha=0.9, 
          sample_rate=0.001,
          n_ins=28 * 28,
          n_outs=10):
    #data_dir, data_file = os.path.split(dataset)
    #output =  data_file[0:-7] + '_zA' + '_alpha' + str(alpha)  + '.pkl.gz'
    #print >> sys.stderr, ("TTTTest output name: ", output)
    #output_folder=data_file[0:-7] + '_pkls'
    #if not os.path.isdir(output_folder):
    #    os.makedirs(output_folder)
    #    print >> sys.stderr, ("make dir: ", output_folder)
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x,  test_set_y  = datasets[2]
    #n_len = train_set_x.get_value(borrow=True).shape[0]
    #print >> sys.stderr, ('lenght of train data: ', n_len)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= pretrain_batch_size
    numpy_rng = numpy.random.RandomState(89677)
    srng=RandomStreams(601)
    print '... building the model'
    deep = deep_sugar(numpy_rng=numpy_rng, n_ins=n_ins,
                      alpha=alpha, sample_rate=sample_rate,
                      hidden_layers_sizes=hidden_layers_sizes,
                      n_outs=n_outs, 
                      allX=train_set_x, allY=train_set_y, 
                      srng=srng)
    
    print '... getting the pretraining functions'
    pretraining_fns = deep.pretraining_functions(train_set_x=train_set_x,
                                                train_set_y=train_set_y,
                                                batch_size=pretrain_batch_size)
    print '... pre-training the model'
    start_time = time.clock()
    corruption_levels = [.1, .2, .3, .4]
    #os.chdir(output_folder)
    for i in xrange(deep.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            tic = time.clock()
            for batch_index in xrange(n_train_batches):           
                switch = 1 if batch_index == 0 else 0
                c0 = pretraining_fns[i](index=batch_index,
                                        corruption=corruption_levels[i],
                                        lr=pretrain_lr,
                                        switch=switch)
                c.append(c0)
                #if switch == 1:
                #    print >> sys.stderr, 'time: %f sec\t epoch %d batch %d switch %d' % ((time.clock() - tic), epoch, batch_index, switch)
            print >> sys.stderr, 'Pre-training layer %i, epoch %d, cost %f [%f sec]' % (i, epoch, numpy.mean(c), (time.clock() - tic) )
            #print numpy.mean(c)
            #print >> sys.stderr, 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            #print >> sys.stderr, numpy.mean(c)
            appr_sparse_rate = getNumZero(deep.sugar_layers[i].W.get_value(borrow=True))
            if appr_sparse_rate > 50:
                break
        #ca = sca.sugar_layers[i]
        #output =  data_file[0:-7] + '_zA' + '_alpha' + str(alpha) + '_plr' + str(pretrain_lr) + '_layer' + str(i)+ '_H' + str(ca.n_hidden) + '.pkl.gz'
        #print >> sys.stderr, ('save pkl to the file: ', output)
        #savePm(output=output,
        #W=ca.W.get_value(borrow=True),
        #bvis=ca.b_prime.get_value(borrow=True),
        #bhid=ca.b.get_value(borrow=True))
    #os.chdir('../')
    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                        os.path.split(__file__)[1] +
                        ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    print >> sys.stderr, (batch_size, n_train_batches)
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = deep.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    print '... finetunning the model'
    patience = 10 * n_train_batches   
    patience_increase = 2.  
    improvement_threshold = 0.995   
    validation_frequency = min(n_train_batches, patience / 2)
    #best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()
    done_looping = False
    epoch = 0
    while (epoch < training_epochs) and (not done_looping):
        tic = time.clock()
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                if this_validation_loss < best_validation_loss:
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    #best_iter = iter
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    #print >> sys.stderr, (('     epoch %i, minibatch %i/%i, test error of '
                    #       'best model %f %%') %
                    #      (epoch, minibatch_index + 1, n_train_batches,
                    #       test_score * 100.))
            #if patience <= iter:
            #    done_looping = True
            #    break
        print >> sys.stderr, 'Fine-tuning, epoch %d [%f sec]' % (epoch, (time.clock() - tic) )

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
if __name__ == '__main__':
    train()
