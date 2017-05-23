import os
import tensorflow as tf
import numpy as np
import math
from ops import encoder, decoder,decoder_vanilla
from generator import Generator

class VAE(Generator):

    def __init__(self, hidden_size, batch_size, learning_rate, channel, model_name):
        self.working_directory = '/tempspace/hyuan/VAE'
        self.height = 64
        self.width = 64                           
        self.modeldir = './modeldir'
        self.logdir = './logdir'
        self.hidden_size = hidden_size
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate =learning_rate
        self.channel = channel
        self.input_tensor =  tf.placeholder(
            tf.float32, [None,  self.height, self.width, 3])
        if not os.path.exists(self.modeldir):
            os.makedirs(self.modeldir)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.configure_networks()
    
    def configure_networks(self):
        with tf.variable_scope('VAE') as scope:
            self.train_summary = self.build_network('train',  self.input_tensor, self.model_name)
        self.writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)

    def build_network(self, name, input_tensor, model_name):
        summarys = []
        with tf.variable_scope('model') as scope:
            if model_name == 'low_rank':
                encode_out = encoder(input_tensor, self.hidden_size*4*self.channel)   #get latent representation from encoder 
            else:
                encode_out = encoder(input_tensor, 2*self.channel)
            new_sample = self.get_sample(encode_out, model_name)

            if model_name == 'low_rank':
                out_put = decoder(new_sample)
            else:
                out_put = decoder_vanilla(new_sample)

        with tf.variable_scope("model", reuse=True) as scope: # use current model to generate images
            if model_name == 'low_rank':
                test_sample= tf.random_normal([self.batch_size,self.channel, self.hidden_size*self.hidden_size])
                test_sample = tf.reshape(test_sample, [self.batch_size, self.channel,self.hidden_size, self.hidden_size])
                self.sample_out = decoder(test_sample)
            else:
                test_sample= tf.random_normal([self.batch_size,self.channel]) 
                self.sample_out = decoder_vanilla(test_sample)   

        self.kl_loss = self.get_loss(self.new_mean,self.new_std)
        self.rec_loss = self.get_rec_loss(out_put, input_tensor)
        total_loss = self.kl_loss + self.rec_loss
        summarys.append(tf.summary.scalar('/KL-loss', self.kl_loss))
        summarys.append(tf.summary.scalar('/Rec-loss', self.rec_loss))
        summarys.append(tf.summary.scalar('/loss', total_loss))
        summarys.append(tf.summary.image('input', tf.reshape(input_tensor, [-1, self.height, self.width, 3]), max_outputs = 20))
        summarys.append(tf.summary.image('output', tf.reshape(out_put, [-1, self.height, self.width, 3 ]), max_outputs = 20))
        self.train = tf.contrib.layers.optimize_loss(total_loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.learning_rate, optimizer='Adam', update_ops=[])
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge(summarys)
        return summary

    
    def get_sample(self, encode_out, model_name):
        if model_name == 'low_rank':
            encode_out = tf.reshape(encode_out, [self.batch_size ,self.channel, 4*self.hidden_size]) #reshape to have u1, sigma1, u2, sigma2 for each channel
            mean1 = encode_out[ :, : , :self.hidden_size] #10*128*d*1
            stddev1 = tf.sqrt(tf.exp(encode_out[:,:,self.hidden_size:2*self.hidden_size]))
            mean2 = encode_out[:,:,2*self.hidden_size:3*self.hidden_size]
            stddev2 = tf.sqrt(tf.exp(encode_out[:,:,3*self.hidden_size:4*self.hidden_size])) #using sqrt of exp to avoid negative
            new_mean= tf.expand_dims(mean1, -1) * tf.expand_dims(mean2, -2)  # low rank formulation
            new_std = tf.expand_dims(stddev1, -1) *tf.expand_dims(stddev2, -2)
            new_mean = tf.reshape(new_mean, [self.batch_size, self.channel, self.hidden_size * self.hidden_size])
            new_std = tf.reshape(new_std, [self.batch_size, self.channel, self.hidden_size * self.hidden_size])
            epsilon = tf.random_normal([self.batch_size, self.channel, self.hidden_size*self.hidden_size])
            new_sample = new_mean + epsilon*new_std          # reparameterization trick
            new_sample = tf.reshape(new_sample,[self.batch_size, self.channel, self.hidden_size, self.hidden_size] )
            self.new_mean= new_mean
            self.new_std =new_std
        elif model_name == 'vanilla':
            mean = encode_out[:,:self.channel]
            stddev = tf.sqrt(tf.exp(encode_out[:,self.channel:]))
            self.new_mean= mean
            self.new_std= stddev
            first_sample = tf.random_normal([self.batch_size,self.channel])
            new_sample = first_sample*stddev + mean
        return new_sample


    def get_loss(self, mean, stddev, epsilon=1e-8):   #KL loss
        return tf.reduce_mean(0.5*(tf.square(mean)+
            tf.square(stddev)-2.0*tf.log(stddev+epsilon)-1.0))

    def get_rec_loss(self, out_put, target_out):      # reconstruction loss

        # return tf.reduce_sum(tf.squared_difference(out_put, target_out))
        return tf.losses.mean_squared_error(out_put, target_out)

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.modeldir, 'model')
        self.saver.save(self.sess, checkpoint_path, global_step=step)
    
    def update_params(self, input_tensor, step):
        loss, summary, kl_loss, rec_loss =  self.sess.run([self.train, self.train_summary, self.kl_loss, self.rec_loss], {self.input_tensor: input_tensor})
        self.writer.add_summary(summary, step)
        return loss, kl_loss, rec_loss

    def reload(self, epoch):
        checkpoint_path = os.path.join(
            self.modeldir, 'model')
        model_path = checkpoint_path +'-'+str(epoch)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return       
        self.saver.restore(self.sess, model_path)
        print("model load successfully===================")
   # def evaluate
    def log_marginal_likelihood_estimate(self):
        x_mean = tf.reshape(self.input_tensor, [self.batch_size, self.width*self.height])
        x_sample = tf.reshape(self.out_put, [self.batch_size,self.width*self.height])
        x_sigma = tf.multiply(1.0, tf.ones(tf.shape(x_mean)))
        return log_likelihood_gaussian(x_mean, x_sample, x_sigma)+\
                log_likelihood_prior(self.latent_sample)-\
                log_likelihood_gaussian(self.latent_sample, self.mean, self.stddev)        

    def evaluate(self, test_input):
        sample_ll= []
        for j in range (1000):
            res= self.sess.run(self.lle,{self.input_tensor: test_input})
            sample_ll.append(res)
        sample_ll = np.array(sample_ll)
        m = np.amax(sample_ll, axis=1, keepdims=True)
        log_marginal_estimate = m + np.log(np.mean(np.exp(sample_ll - m), axis=1, keepdims=True))
        return np.mean(log_marginal_estimate)

    def generate_samples(self):
        samples = []
        for i in range(100): # generate 100*100 samples
            samples.extend(self.sess.run(self.sample_out))
        samples = np.array(samples)
        print (samples.shape)
        return samples



        

