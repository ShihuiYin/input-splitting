
from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234) # for reproducibility
import argparse
from ast import literal_eval as bool

import lasagne
import theano
import theano.tensor as T
sys.setrecursionlimit(50000)
import scipy.io as sio
import binary_net

from pylearn2.datasets.cifar10 import CIFAR10 

from collections import OrderedDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR10 with input-splitting", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-bs', dest='batch_size', type=int, default=100, help='batch size')
    parser.add_argument('-sp', dest='save_path', default='cifar10_parameters.npz', help='path to saved model')
    parser.add_argument('-ne', dest='num_epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('-wp', dest='weight_prec', type=int, default=1, help='weight precision')
    parser.add_argument('-ls', dest='LR_start', type=float, default=0.005, help='start learning rate')
    parser.add_argument('-lf', dest='LR_finish', type=float, default=3e-7, help='finish learning rate')
    parser.add_argument('-tr', dest='train', type=bool, default=True, help='training if true, else testing')
    parser.add_argument('-st', dest='stochastic', type=bool, default=False, help='stochastic')
    parser.add_argument('-nr', dest='num_rows', type=int, default=64, help='number of rows for mapping')
    parser.add_argument('-mc', dest='monte_carlo', type=int, default=1, help='number of monte carlo runs')
    parser.add_argument('-rf', dest='result_file', default='cifar10_test_error_list.mat', help='path to saved results')
    parser.add_argument('-sz', dest='size', default='heavy', help='model size (heavy or light)')
    parser.add_argument('-sw', dest='sim_weight_variation', type=bool, default=False, help='simulate weight variation if true')
    parser.add_argument('-an', dest='act_noise', type=float, default=0.0, help='activation noise')
    
    args = parser.parse_args()
    
    print (args)
    # BN parameters
    weight_prec = args.weight_prec
    act_noise = args.act_noise
    batch_size = args.batch_size
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    save_path = args.save_path  
    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")
    num_rows = args.num_rows
    # BinaryConnect    
    binary = args.train
    print("binary = "+str(binary))
    stochastic = args.stochastic
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.    
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))
    
    # Training parameters
    num_epochs = args.num_epochs
    print("num_epochs = "+str(num_epochs))
    
    # Decaying LR 
    LR_start = args.LR_start
    print("LR_start = "+str(LR_start))
    LR_fin = args.LR_finish
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    train_set_size = 45000
    print("train_set_size = "+str(train_set_size))
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    
    print('Loading CIFAR-10 dataset...')
    
    if args.train:
        train_set = CIFAR10(which_set="train",start=0,stop = train_set_size)
        valid_set = CIFAR10(which_set="train",start=train_set_size,stop = 50000)
    test_set = CIFAR10(which_set="test")
        
    # bc01 format
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
    if args.train:
        train_set.X = np.reshape(np.subtract(np.multiply(2./255.,train_set.X),1.),(-1,3,32,32))
        valid_set.X = np.reshape(np.subtract(np.multiply(2./255.,valid_set.X),1.),(-1,3,32,32))
    test_set.X = np.reshape(np.subtract(np.multiply(2./255.,test_set.X),1.),(-1,3,32,32))
    
    # flatten targets
    if args.train:
        train_set.y = np.hstack(train_set.y)
        valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)
    
    # Onehot the targets
    if args.train:
        train_set.y = np.float32(np.eye(10)[train_set.y])    
        valid_set.y = np.float32(np.eye(10)[valid_set.y])
    test_set.y = np.float32(np.eye(10)[test_set.y])
    
    # for hinge loss
    if args.train:
        train_set.y = 2* train_set.y - 1.
        valid_set.y = 2* valid_set.y - 1.
    test_set.y = 2* test_set.y - 1.

    print('Building the CNN...') 
    if args.size == 'heavy':
        sizes = [126, 126, 252, 252, 504, 512, 1024, 1024, 10]
    elif args.size == 'light':
        sizes = [63, 63, 126, 126, 252, 256, 512, 512, 10]
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    cnn = lasagne.layers.InputLayer(
            shape=(None, 3, 32, 32),
            input_var=input)
    
    # 128C3-128C3-P2             
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            weight_prec=weight_prec,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=sizes[0], 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
            
    cnn = binary_net.Conv2DLayer_Fanin_Limited(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            weight_prec=weight_prec,
            H=H,
            W_LR_scale=W_LR_scale,
            max_fan_in=num_rows,
            num_filters=sizes[1], 
            filter_size=(3, 3),
            pad=1,
            act_noise=act_noise,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
            
    # 256C3-256C3-P2             
    cnn = binary_net.Conv2DLayer_Fanin_Limited(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            weight_prec=weight_prec,
            H=H,
            W_LR_scale=W_LR_scale,
            max_fan_in=num_rows,
            num_filters=sizes[2], 
            filter_size=(3, 3),
            pad=1,
            act_noise=act_noise,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
            
    cnn = binary_net.Conv2DLayer_Fanin_Limited(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            weight_prec=weight_prec,
            H=H,
            W_LR_scale=W_LR_scale,
            max_fan_in=num_rows,
            num_filters=sizes[3], 
            filter_size=(3, 3),
            pad=1,
            act_noise=act_noise,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
            
    # 512C3-512C3-P2              
    cnn = binary_net.Conv2DLayer_Fanin_Limited(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            weight_prec=weight_prec,
            H=H,
            W_LR_scale=W_LR_scale,
            max_fan_in=num_rows,
            num_filters=sizes[4], 
            filter_size=(3, 3),
            pad=1,
            act_noise=act_noise,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
                  
    cnn = binary_net.Conv2DLayer_Fanin_Limited(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            weight_prec=weight_prec,
            H=H,
            W_LR_scale=W_LR_scale,
            max_fan_in=num_rows,
            num_filters=sizes[5], 
            filter_size=(3, 3),
            pad=1,
            act_noise=act_noise,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
    
    # print(cnn.output_shape)
    
    # 1024FP-1024FP-10FP            
    cnn = binary_net.DenseLayer_Fanin_Limited(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                weight_prec=weight_prec,
                H=H,
                W_LR_scale=W_LR_scale,
                max_fan_in=num_rows,
                nonlinearity=lasagne.nonlinearities.identity,
                name='FC1',
                act_noise=act_noise,
                num_units=sizes[6])      
                  
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
            
    cnn = binary_net.DenseLayer_Fanin_Limited(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                weight_prec=weight_prec,
                H=H,
                W_LR_scale=W_LR_scale,
                max_fan_in=num_rows,
                nonlinearity=lasagne.nonlinearities.identity,
                name='FC2',
                act_noise=act_noise,
                num_units=sizes[7])      
                  
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
    
    cnn = binary_net.DenseLayer_Fanin_Limited(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                weight_prec=weight_prec,
                H=H,
                W_LR_scale=W_LR_scale,
                max_fan_in=num_rows,
                nonlinearity=lasagne.nonlinearities.identity,
                name='FC3',
                act_noise=act_noise,
                num_units=10)      
                  
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)

    train_output = lasagne.layers.get_output(cnn, deterministic=False)
    
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    if args.train:
        if binary:
            
            # W updates
            W = lasagne.layers.get_all_params(cnn, binary=True)
            W_grads = binary_net.compute_grads(loss,cnn)
            updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
            updates = binary_net.clipping_scaling(updates,cnn) # weight scaling disabled
            
            # other parameters updates
            params = lasagne.layers.get_all_params(cnn, trainable=True, binary=False)
            updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
            
        else:
            params = lasagne.layers.get_all_params(cnn, trainable=True)
            updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)
        
        train_fn = theano.function([input, target, LR], loss, updates=updates)
        
    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:

    # Compile a second function computing the validation loss and accuracy:
    print("Compiling...")
    val_fn = theano.function([input, target], [test_loss, test_err])

    if args.train:
        print('Training...')
        binary_net.train(
                train_fn,val_fn,
                cnn,
                batch_size,
                LR_start,LR_decay,
                num_epochs,
                train_set.X,train_set.y,
                valid_set.X,valid_set.y,
                test_set.X,test_set.y,
                save_path,
                shuffle_parts=shuffle_parts)
    else:
        print("Loading the trained parameters and binarizing the weights...")
    
        # Load parameters
        with np.load(save_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(cnn, param_values)

        
        # Binarize the weights
        params = lasagne.layers.get_all_params(cnn)
        for param in params:
            print(param.name)
            if param.name[-1] == "W":
                if weight_prec == 1:
                    if args.sim_weight_variation:
                        W_bin = binary_net.SignNumpy(param.get_value())
                        W_var = W_bin * np.random.normal(loc=1.0, scale=0.02, size=W_bin.shape).astype('float32')
                        param.set_value(W_var)
                    else:
                        param.set_value(binary_net.SignNumpy(param.get_value()))
                elif weight_prec == 2:
                    if args.sim_weight_variation:
                        W_quat = binary_net.QuaternaryNumpy(param.get_value())
                        W_var = W_quat * (abs(W_quat) == 1./3).astype('float32') * np.random.normal(loc=1.0, scale=0.2, size=W_quat.shape)
                        W_var += W_quat * (abs(W_quat) == 1.0).astype('float32') * np.random.normal(loc=1.0, scale=0.04, size=W_quat.shape)
                        param.set_value(W_var.astype('float32'))
                    else:
                        param.set_value(binary_net.QuaternaryNumpy(param.get_value()))
                print(param.get_value())
        print('Running...')
        
        test_error_list = []
        start_time = time.time()
        num_images = 10000
        num_batches = int(num_images / batch_size)
        for r in range(args.monte_carlo):
            test_error = 0.
            for i in range(num_batches):
                print(" batch%d/%d" %(i+1, num_batches))
                # import pdb; pdb.set_trace()
                l, e = val_fn(test_set.X[(i*batch_size):(i+1)*batch_size],test_set.y[(i*batch_size):(i+1)*batch_size])
                test_error += e * 100.
            test_error = test_error / num_batches
            print("Run %d:test_error = %.2f%%" % (r, test_error))
            test_error_list.append(test_error)
            run_time = time.time() - start_time
            print("run_time = "+str(run_time)+"s")
        dict = {"test_error_list": np.array(test_error_list)}
        sio.savemat(args.result_file, dict)
