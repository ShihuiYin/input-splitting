
import time

from collections import OrderedDict

import numpy as np

import lasagne
# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T


from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex, floor, ceil, minimum, sqr, round_half_away_from_zero
from theano.tensor.elemwise import Elemwise

# Our own rounding function, that does not set the gradient to 0 like Theano's
class Round3(UnaryScalarOp):
    
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = round(%(x)s);" % locals()
    
    def grad(self, inputs, gout):
        (gz,) = gout
        return gz, 
        
round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)

# Endy By Shihui, 2/13/2018
def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

def hard_quaternarization(x):
    return (T.round(T.clip(x, -1.0, 1.0) * 1.5 + 1.5) - 1.5) / 1.5

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1 
# during back propagation
def binary_tanh_unit(x):
    return 2.*round3(hard_sigmoid(x))-1.


def binary_activation(x):
    return round3(T.clip(x/2., 0, 1))

import scipy.io as sio
#data = sio.loadmat('prob_chip14_column0_24k_single_SA_0p8V.mat') # for 1-bit weight
# Below are default prob table settings, overwrite it in the main script for other settings
data = sio.loadmat('prob_2bW_chip14.mat') # for 2-bit weight
prob = data['prob'][:,0].astype('float32')
num_rows = prob.shape[0] - 1
prob = theano.shared(prob, name='prob', borrow=True)
levels = np.array([-1., 1.], dtype='float32')
levels = theano.shared(levels, name='levels', borrow=True)
levels_val = levels.get_value()
num_levels = len(levels_val)

def stochastic_round1(x, srng):
    x_int = T.cast(x / 2 + num_rows / 2, 'int32')
    prob_x = T.cast(srng.binomial(n=1, p=prob[x_int], size=T.shape(x)), theano.config.floatX) - 1.
    return T.cast(T.switch(prob_x,-1.,1.), theano.config.floatX)
    
def stochastic_round(x, srng):
    x_int = T.cast(x / 2 + num_rows / 2, 'int32')
    prob_x = 2. * T.cast(T.ge(prob[x_int], srng.uniform(low=0., high=1., size=T.shape(x))), theano.config.floatX) - 1.
    return prob_x

def quant_stoch(x, srng):
    '''
    Stochastically quantize x according to the prob table and quantization levels defined above
    '''
    x_int = T.cast(x / 2 + num_rows / 2, 'int32')
    x_cdf = prob[x_int, 0:num_levels-1].cumsum(axis=-1)
    x_rand = srng.uniform(low=0., high=1., size=T.shape(x))
    x_rand = T.stack([x_rand] * (num_levels-1), axis=-1)
    x_comp = T.cast(T.ge(x_rand, x_cdf), 'int32').sum(axis=-1)
    y = levels[x_comp]
    return y

def quant_det(x, srng=None):
    '''
    Stochastically quantize x according to the prob table and quantization levels defined above
    '''
    x_comp = 0
    for i in range(num_levels - 1):
        x_comp += T.cast(T.ge(x, edges[i]), 'int32')
    y = levels[x_comp]
    return y

quant = quant_det

def get_ideal_quant_prob_levels(num_levels, first_level, level_interval, lower_bound, step_size=2):
    last_level = first_level + level_interval * (num_levels - 1)
    levels = np.array([x for x in range(first_level, last_level+1, level_interval)], dtype='float32')
    edges = (levels[0:num_levels-1] + levels[1:num_levels]) / 2.
    num_rows = 2 * int(-lower_bound / step_size) + 1
    prob = np.zeros((num_rows, num_levels), dtype='float32')
    entries = np.array([x for x in range(lower_bound, -lower_bound+1, step_size)], dtype='float32')
    entries_int = np.array([x for x in range(entries.shape[0])], dtype='int32')
    indices = np.zeros_like(entries).astype('int32')
    for i in range(num_levels - 1):
        indices += (entries > edges[i]).astype('int32')
    prob[entries_int, indices] = 1.
    return prob, levels
    
def binary_tanh_unit_wide(x, deterministic=False, stochastic=False, srng=None):
    if not deterministic:
        return binary_tanh_unit(x/20.)
    if not stochastic:
        return binary_tanh_unit(x/20.)
    else:
        return stochastic_round(x, srng)

    
def SignNumpy(x):
    return np.float32(2.*np.greater_equal(x,0)-1.)
def QuaternaryNumpy(x):
    return ((np.round(np.clip(x, -1.0, 1.0) * 1.5 + 1.5) - 1.5) / 1.5).astype('float32')
#The weights' binarization function, 
# taken directly from the BinaryConnect github repository 
# (which was made available by his authors)
def binarization(W,H,binary=True,deterministic=False,stochastic=False,srng=None):
    
    # (deterministic == True) <-> test-time <-> inference-time
    if not binary or (deterministic and stochastic):
        print("not binary")
        Wb = W
    else:
        
        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)
        # Wb = T.clip(W/H,-1,1)
        
        # Stochastic BinaryConnect
        if stochastic:
        
            # print("stoch")
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)

        # Deterministic BinaryConnect (round to nearest)
        else:
            # print("det")
            Wb = T.round(Wb)
        
        # 0 or 1 -> -1 or 1
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)
    
    return Wb

def quaternarization(W,H,binary=True,deterministic=False,stochastic=False,srng=None):
    
    # (deterministic == True) <-> test-time <-> inference-time
    if not binary or (deterministic and stochastic):
        Wb = W
    else:
        Wb = hard_quaternarization(W)
    return Wb

# This class extends the Lasagne DenseLayer to support BinaryConnect
class DenseLayer(lasagne.layers.DenseLayer):
    
    def __init__(self, incoming, num_units, 
        binary = True, stochastic = True, H=1.,W_LR_scale="Glorot", weight_prec=1, 
        max_fan_in = -1, act_noise=0, **kwargs):
        
        self.binary = binary
        self.stochastic = stochastic
        
        self.H = H
        self.weight_prec = weight_prec
        self.act_noise = act_noise
        if H == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.H = np.float32(np.sqrt(1.5/ (num_inputs + num_units)))
            
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))
        self.max_fan_in = max_fan_in
        if self.max_fan_in > 0:
            self.num_groups = int(np.ceil(num_inputs / max_fan_in))
            self.num_inputs = num_inputs

        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        
        if self.binary:
            super(DenseLayer, self).__init__(incoming, num_units, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)
            # add the binary tag to weights            
            self.params[self.W]=set(['binary'])
            
        else:
            super(DenseLayer, self).__init__(incoming, num_units, **kwargs)
        
    def get_output_for(self, input, deterministic=False, **kwargs):
        if self.weight_prec == 1:
            self.Wb = binarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        elif self.weight_prec == 2:
            self.Wb = quaternarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)

        Wr = self.W
        self.W = self.Wb
        if self.max_fan_in > 0:
            num_leading_axes = self.num_leading_axes
            if num_leading_axes < 0:
                num_leading_axes += input.ndim
            if input.ndim > num_leading_axes + 1:
                input = input.flatten(num_leading_axes + 1)
            
            rvalue = 0
            for i in range(self.num_groups):
                start_index = i * self.max_fan_in
                stop_index = np.minimum((i+1)*self.max_fan_in, self.num_inputs)
                partial_sum = T.dot(input[:,start_index:stop_index], self.W[start_index:stop_index,:])
                if self.weight_prec == 2:
                    partial_sum *= 3.
                if self.act_noise > 0:
                    partial_sum += self._srng.normal(partial_sum.shape, avg=0.0, std=self.act_noise)
                rvalue += quant(partial_sum, self._srng)
            if self.weight_prec == 2:
                rvalue /= 3.
            rvalue += self.b
            rvalue = self.nonlinearity(rvalue)
        else:
            rvalue = super(DenseLayer, self).get_output_for(input, **kwargs)
        
        self.W = Wr
        
        return rvalue
        
# This class extends the Lasagne DenseLayer to support BinaryConnect
class DenseLayer_Fanin_Limited(lasagne.layers.DenseLayer):
    
    def __init__(self, incoming, num_units, 
        binary = True, stochastic = True, H=1.,W_LR_scale="Glorot",
        max_fan_in = 64, weight_prec=1, act_noise=0, **kwargs):
        
        self.binary = binary
        self.stochastic = stochastic
        self.max_fan_in = max_fan_in
        self.weight_prec = weight_prec
        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.H = np.float32(np.sqrt(1.5/ (num_inputs + num_units)))
            
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))
        num_inputs = int(np.prod(incoming.output_shape[1:]))
        self.num_groups = int(np.ceil(num_inputs / max_fan_in))
        self.num_inputs = num_inputs
        self.act_noise = act_noise
        
        

        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        
        if self.binary:
            super(DenseLayer_Fanin_Limited, self).__init__(incoming, num_units, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)
            # add the binary tag to weights            
            self.params[self.W]=set(['binary'])
            
        else:
            super(DenseLayer_Fanin_Limited, self).__init__(incoming, num_units, **kwargs)
        scale = np.float32(1./np.sqrt(1.5/(self.num_groups+num_units)))
        # self.W_group = self.add_param(lasagne.init.Uniform((-scale, scale)), (self.num_groups,), name="W_group")
    def get_output_for(self, input, deterministic=False, **kwargs):
        if self.weight_prec == 1:
            self.Wb = binarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        elif self.weight_prec == 2:
            self.Wb = quaternarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        Wr = self.W
        self.W = self.Wb
        
        num_leading_axes = self.num_leading_axes
        if num_leading_axes < 0:
            num_leading_axes += input.ndim
        if input.ndim > num_leading_axes + 1:
            input = input.flatten(num_leading_axes + 1)
        
        rvalue = 0
        for i in range(self.num_groups):
            start_index = i * self.max_fan_in
            stop_index = np.minimum((i+1)*self.max_fan_in, self.num_inputs)
            partial_sum = T.dot(input[:,start_index:stop_index], self.W[start_index:stop_index,:])
            if self.act_noise > 0:
                partial_sum += self._srng.normal(partial_sum.shape, avg=0.0, std=self.act_noise)
            if deterministic and self.weight_prec == 2:
                rvalue += binary_tanh_unit_wide(partial_sum*3, deterministic, self.stochastic, self._srng) * 4. # * self.W_group[i]
            else:
                rvalue += binary_tanh_unit_wide(partial_sum, deterministic, self.stochastic, self._srng) * 4. # * self.W_group[i]
            #rvalue += T.dot(input[:,start_index:stop_index], self.W[start_index:stop_index,:]) # * self.W_group[i]
        rvalue += self.b
        rvalue = self.nonlinearity(rvalue)
        self.W = Wr
        
        return rvalue

# This class extends the Lasagne Conv2DLayer to support BinaryConnect
class Conv2DLayer(lasagne.layers.Conv2DLayer):
    
    def __init__(self, incoming, num_filters, filter_size,
        binary = True, stochastic = True, H=1.,W_LR_scale="Glorot", weight_prec=1, 
        max_fan_in=64, act_noise=0, **kwargs):
        
        self.binary = binary
        self.stochastic = stochastic
        
        self.H = H
        self.weight_prec = weight_prec
        if H == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))
        
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.W_LR_scale = np.float32(1./np.sqrt(1.5 / (num_inputs + num_units)))
        self.max_fan_in = max_fan_in
        if self.max_fan_in > 0:
            self.num_channels_per_array = int(np.floor(max_fan_in / np.prod(filter_size).astype('float32')))
            self.num_arrays = int(np.ceil(incoming.output_shape[1] / self.num_channels_per_array))
            self.num_channels = incoming.output_shape[1]
            print("Number of arrays: %d" % self.num_arrays)
            
        self.act_noise = act_noise
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
            
        if self.binary:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)   
            # add the binary tag to weights            
            self.params[self.W]=set(['binary'])
        else:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)    
    
    def convolve(self, input, deterministic=False, **kwargs):
        if self.weight_prec == 1:
            self.Wb = binarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        elif self.weight_prec == 2:
            self.Wb = quaternarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        Wr = self.W
        self.W = self.Wb
        if self.max_fan_in > 0:            
            border_mode = 'half' if self.pad == 'same' else self.pad
            rvalue = 0
            for i in range(self.num_arrays):
                start_index = i * self.num_channels_per_array
                stop_index = np.minimum((i+1)*self.num_channels_per_array, self.num_channels)
                input_shape = (self.input_shape[0], stop_index - start_index, self.input_shape[2], self.input_shape[3])
                W_shape = self.get_W_shape()
                W_shape_new = (W_shape[0], stop_index - start_index, W_shape[2], W_shape[3])
                partial_sum = self.convolution(input[:,start_index:stop_index,:,:], self.W[:,start_index:stop_index,:,:],
                            input_shape, W_shape_new, subsample=self.stride, border_mode=border_mode, filter_flip=self.flip_filters, **kwargs)
                if self.weight_prec == 2:
                    partial_sum *= 3.
                if self.act_noise > 0:
                    partial_sum += self._srng.normal(partial_sum.shape, avg=0.0, std=self.act_noise)
                rvalue += quant(partial_sum, self._srng)
            if self.weight_prec == 2:
                rvalue /= 3.
        else:
            rvalue = super(Conv2DLayer, self).convolve(input, **kwargs)
        
        self.W = Wr
        
        return rvalue
        
class Conv2DLayer_Fanin_Limited(lasagne.layers.Conv2DLayer):
    
    def __init__(self, incoming, num_filters, filter_size, max_fan_in=64,
        binary=True, stochastic=True, H=1.,W_LR_scale="Glorot", weight_prec=1, act_noise=0, **kwargs):
        
        self.binary = binary
        self.stochastic = stochastic
        self.max_fan_in = max_fan_in
        self.H = H
        self.weight_prec = weight_prec
        if H == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))
        self.num_channels_per_array = int(np.floor(max_fan_in / np.prod(filter_size).astype('float32')))
        self.num_arrays = int(np.ceil(incoming.output_shape[1] / self.num_channels_per_array))
        self.num_channels = incoming.output_shape[1]
        print("Number of arrays: %d" % self.num_arrays)
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.W_LR_scale = np.float32(1./np.sqrt(1.5 / (num_inputs + num_units)))
            
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        self.act_noise = act_noise
            
        if self.binary:
            super(Conv2DLayer_Fanin_Limited, self).__init__(incoming, num_filters, filter_size, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)   
            # add the binary tag to weights            
            self.params[self.W]=set(['binary'])
        else:
            super(Conv2DLayer_Fanin_Limited, self).__init__(incoming, num_filters, filter_size, **kwargs)    
        # scale = np.float32(1./np.sqrt(1.5/self.num_groups))
        # self.W_group = self.add_param(lasagne.init.Uniform((-scale, scale)), (self.num_groups,), name="W_group")
    def convolve(self, input, deterministic=False, **kwargs):
        if self.weight_prec == 1:
            self.Wb = binarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        elif self.weight_prec == 2:
            self.Wb = quaternarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        Wr = self.W
        self.W = self.Wb
        
        border_mode = 'half' if self.pad == 'same' else self.pad
        convolved = 0
        for i in range(self.num_arrays):
            start_index = i * self.num_channels_per_array
            stop_index = np.minimum((i+1)*self.num_channels_per_array, self.num_channels)
            input_shape = (self.input_shape[0], stop_index - start_index, self.input_shape[2], self.input_shape[3])
            W_shape = self.get_W_shape()
            W_shape_new = (W_shape[0], stop_index - start_index, W_shape[2], W_shape[3])
            partial_sum = self.convolution(input[:,start_index:stop_index,:,:], self.W[:,start_index:stop_index,:,:],
                        input_shape, W_shape_new, subsample=self.stride, border_mode=border_mode, filter_flip=self.flip_filters, **kwargs)
            if self.act_noise > 0:
                partial_sum += self._srng.normal(partial_sum.shape, avg=0.0, std=self.act_noise)
            if deterministic and self.weight_prec == 2:
                convolved += binary_tanh_unit_wide(partial_sum * 3, deterministic, self.stochastic, self._srng) * 4.  # * self.W_group[i]
            else:
                convolved += binary_tanh_unit_wide(partial_sum, deterministic, self.stochastic, self._srng) * 4.  # * self.W_group[i]
            #convolved += self.convolution(input[:,start_index:stop_index,:,:], self.W[:,start_index:stop_index,:,:],
            #            input_shape, W_shape_new, subsample=self.stride, border_mode=border_mode, filter_flip=self.flip_filters, **kwargs)  # * self.W_group[i]
        self.W = Wr
        
        return convolved

# This function computes the gradient of the binary weights
def compute_grads(loss,network):
        
    layers = lasagne.layers.get_all_layers(network)
    grads = []
    
    for layer in layers:
    
        params = layer.get_params(binary=True)
        if params:
            grads.append(theano.grad(loss, wrt=layer.Wb))
                
    return grads

# This functions clips the weights after the parameter update
def clipping_scaling(updates,network):
    
    layers = lasagne.layers.get_all_layers(network)
    updates = OrderedDict(updates)
    
    for layer in layers:
    
        params = layer.get_params(binary=True)
        for param in params:
            print("W_LR_scale = "+str(layer.W_LR_scale))
            print("H = "+str(layer.H))
            updates[param] = param + layer.W_LR_scale*(updates[param] - param)
            updates[param] = T.clip(updates[param], -layer.H,layer.H)     

    return updates
        
# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default trainer function in Lasagne yet)
def train(train_fn,val_fn,
            model,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            X_train,y_train,
            X_val,y_val,
            X_test,y_test,
            save_path=None,
            shuffle_parts=1):
    
    # A function which shuffles a dataset
    def shuffle(X,y):
        
        
        chunk_size = len(X)/shuffle_parts
        shuffled_range = range(chunk_size)
        
        X_buffer = np.copy(X[0:chunk_size])
        y_buffer = np.copy(y[0:chunk_size])
        
        for k in range(shuffle_parts):
            
            np.random.shuffle(shuffled_range)

            for i in range(chunk_size):
                
                X_buffer[i] = X[k*chunk_size+shuffled_range[i]]
                y_buffer[i] = y[k*chunk_size+shuffled_range[i]]
            
            X[k*chunk_size:(k+1)*chunk_size] = X_buffer
            y[k*chunk_size:(k+1)*chunk_size] = y_buffer
        
        return X,y
        
        # shuffled_range = range(len(X))
        # np.random.shuffle(shuffled_range)
        
        # new_X = np.copy(X)
        # new_y = np.copy(y)
        
        # for i in range(len(X)):
            
            # new_X[i] = X[shuffled_range[i]]
            # new_y[i] = y[shuffled_range[i]]
            
        # return new_X,new_y
    
    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X,y,LR):
        
        loss = 0
        batches = len(X)/batch_size
        
        for i in range(batches):
            loss += train_fn(X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size],LR)
        
        loss/=batches
        
        return loss
    
    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X,y):
        
        err = 0
        loss = 0
        batches = len(X)/batch_size
        
        for i in range(batches):
            new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            err += new_err
            loss += new_loss
        
        err = err / batches * 100
        loss /= batches

        return err, loss
    
    # shuffle the train set
    X_train,y_train = shuffle(X_train,y_train)
    best_val_err = 100
    best_epoch = 1
    LR = LR_start
    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        
        start_time = time.time()
        
        train_loss = train_epoch(X_train,y_train,LR)
        X_train,y_train = shuffle(X_train,y_train)
        
        val_err, val_loss = val_epoch(X_val,y_val)
        
        # test if validation error went down
        if val_err <= best_val_err:
            
            best_val_err = val_err
            best_epoch = epoch+1
            
            test_err, test_loss = val_epoch(X_test,y_test)
            
            if save_path is not None:
                np.savez(save_path, *lasagne.layers.get_all_param_values(model))
        
        epoch_duration = time.time() - start_time
        
        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  validation error rate:         "+str(val_err)+"%")
        print("  best epoch:                    "+str(best_epoch))
        print("  best validation error rate:    "+str(best_val_err)+"%")
        print("  test loss:                     "+str(test_loss))
        print("  test error rate:               "+str(test_err)+"%") 
        
        # decay the LR
        LR *= LR_decay
