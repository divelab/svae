import tensorflow as tf
import numpy as np

conv_size = 5
deconv_size_first = 2
deconv_size_second = 3
deconv_size = 5

def encoder(input_tensor, output_size): 
    output = tf.contrib.layers.conv2d(
        input_tensor, 32, conv_size, scope='convlayer1', stride =2, 
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d(
        output, 64, conv_size, scope='convlayer2', stride =2, 
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d(
        output, 128, conv_size, scope='convlayer3', stride =2, padding='VALID',
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True}) 
    output = tf.contrib.layers.dropout(output, 0.9, scope='dropout1')
    output = tf.contrib.layers.flatten(output)
    return tf.contrib.layers.fully_connected(output, output_size, activation_fn=None)

def decoder(input_sensor):
    output = tf.transpose(input_sensor, perm=[0, 2, 3 ,1])
    output = tf.contrib.layers.conv2d_transpose(
        output, 64, deconv_size_second, scope='deconv1', stride = 2,
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d_transpose(
        output, 32, deconv_size_second, scope='deconv2', padding='VALID',
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d_transpose(
        output, 32, deconv_size, scope='deconv3', stride = 2,
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d_transpose(
        output, 16, deconv_size, scope='deconv4', stride = 2,
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d_transpose(
        output, 3, deconv_size, scope='deconv5', stride=2,
        activation_fn=tf.nn.tanh, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})   
    return output



def log_likelihood_gaussian(sample, mean, sigma):
    '''
    compute log(sample~Gaussian(mean, sigma^2))
    '''
    return -log2pi*tf.cast(sample.shape[1].value, tf.float32)/2\
        -tf.reduce_sum(tf.square((sample-mean)/sigma) + 2*tf.log(sigma), 1)/2

def log_likelihood_prior(sample):
    '''
    compute log(sample~Gaussian(0, I))
    '''
    return -log2pi*tf.cast(sample.shape[1].value, tf.float32)/2\
         -tf.reduce_sum(tf.square(sample), 1)/2

def parzen_cpu_batch(x_batch, samples, sigma, batch_size, num_of_samples, data_size):
    '''
    x_batch:    a data batch (batch_size, data_size), data_size = h*w*c for images
    samples:    generated data (num_of_samples, data_size)
    sigma:      standard deviation (float32)
    '''
    x = x_batch.reshape((batch_size, 1, data_size))
    mu = samples.reshape((1, num_of_samples, data_size))
    a = (x - mu)/sigma # (batch_size, num_of_samples, data_size)

    # sum -0.5*a^2
    tmp = -0.5*(a**2).sum(2) # (batch_size, num_of_samples)
    # log_mean_exp trick
    max_ = np.amax(tmp, axis=1, keepdims=True) # (batch_size, 1)
    E = max_ + np.log(np.mean(np.exp(tmp - max_), axis=1, keepdims=True)) # (batch_size, 1)
    # Z = dim * log(sigma * sqrt(2*pi)), dim = data_size
    Z = data_size * np.log(sigma * np.sqrt(np.pi * 2))
    return E-Z