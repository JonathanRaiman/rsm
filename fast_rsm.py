"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""
import time

import numpy
import random

import theano
import theano.tensor as T

from math import floor

from os.path import isfile as file_exists

from theano.tensor.shared_randomstreams import RandomStreams


class RSM(object):
	"""Replicated Softmax Machine (RSM)  """
	def __init__(self, input=None, scaling = None, n_visible=784, n_hidden=500, \
		W=None, hbias=None, vbias=None, numpy_rng=None,
		theano_rng=None):
		"""
		RSM constructor. Defines the parameters of the model along with
		basic operations for inferring hidden from visible (and vice-versa),
		as well as for performing CD updates.

		:param input: None for standalone RBMs or symbolic variable if RBM is
		part of a larger graph.

		:param n_visible: number of visible units

		:param n_hidden: number of hidden units

		:param W: None for standalone RBMs or symbolic variable pointing to a
		shared weight matrix in case RBM is part of a DBN network; in a DBN,
		the weights are shared between RBMs and layers of a MLP

		:param hbias: None for standalone RBMs or symbolic variable pointing
		to a shared hidden units bias vector in case RBM is part of a
		different network

		:param vbias: None for standalone RBMs or a symbolic variable
		pointing to a shared visible units bias
		"""

		self.n_visible = n_visible
		self.n_hidden = n_hidden

		if numpy_rng is None:
			# create a number generator
			numpy_rng = numpy.random.RandomState(1234)

		if theano_rng is None:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		if W is None:
			# W is initialized with `initial_W` which is uniformely
			# sampled from -4*sqrt(6./(n_visible+n_hidden)) and
			# 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
			# converted using asarray to dtype theano.config.floatX so
			# that the code is runable on GPU
			initial_W = numpy.asarray(numpy_rng.uniform(
					  low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
					  high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
					  size=(n_visible, n_hidden)),
					  dtype=theano.config.floatX)
			# theano shared variables for weights and biases
			W = theano.shared(value=initial_W, name='W', borrow=True)

		if hbias is None:
			# create shared variable for hidden units bias
			hbias = theano.shared(value=numpy.zeros(n_hidden,
													dtype=theano.config.floatX),
								  name='hbias', borrow=True)

		if vbias is None:
			# create shared variable for visible units bias
			vbias = theano.shared(value=numpy.zeros(n_visible,
													dtype=theano.config.floatX),
								  name='vbias', borrow=True)

		# initialize input layer for standalone RBM or layer0 of DBN
		self.input = input
		self.scaling = scaling

		if not input:
			self.input = T.matrix('input')
		if not scaling:
			self.scaling = T.vector('scaling')

		self.W = W
		self.hbias = hbias
		self.vbias = vbias
		self.theano_rng = theano_rng
		# **** WARNING: It is not a good idea to put things in this list
		# other than shared variables created in this function.
		self.params = [self.W, self.hbias, self.vbias]

	def free_energy(self, v_sample, scaling):
		''' Function to compute the free energy '''

		# look at fast_rsm.ipynb for explanation.

		# hidden layer activation:
		wx_b = T.dot(v_sample, self.W) + T.outer(scaling, self.hbias)
		
		vbias_term = T.dot(v_sample, self.vbias)
		hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
		return -hidden_term - vbias_term

	def propup(self, vis):
		'''This function propagates the visible units activation upwards to
		the hidden units

		Note that we return also the pre-sigmoid activation of the
		layer. As it will turn out later, due to how Theano deals with
		optimizations, this symbolic variable will be needed to write
		down a more stable computational graph (see details in the
		reconstruction cost function)

		'''
		pre_sigmoid_activation = T.dot(vis, self.W) + T.outer(self.scaling, self.hbias)
		return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

	def sample_h_given_v(self, v0_sample):
		''' This function infers state of hidden units given visible units '''
		# compute the activation of the hidden units given a sample of
		# the visibles
		pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
		# get a sample of the hiddens given their activation
		# Note that theano_rng.binomial returns a symbolic sample of dtype
		# int64 by default. If we want to keep our computations in floatX
		# for the GPU we need to specify to return the dtype floatX
		h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
								 n=1, p=h1_mean,
								 dtype=theano.config.floatX)
		return [pre_sigmoid_h1, h1_mean, h1_sample]

	def propdown(self, hid):
		'''This function propagates the hidden units activation downwards to
		the visible units

		Note that we return also the pre_sigmoid_activation of the
		layer. As it will turn out later, due to how Theano deals with
		optimizations, this symbolic variable will be needed to write
		down a more stable computational graph (see details in the
		reconstruction cost function)

		'''
		pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
		# softmax here, not sigmoid. !!
		return [pre_sigmoid_activation, T.nnet.softmax(pre_sigmoid_activation)]

	def sample_v_given_h(self, h0_sample):
		''' This function infers state of visible units given hidden units '''
		# compute the activation of the visible given the hidden sample
		pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
		# get a sample of the visible given their activation
		# Note that theano_rng.binomial returns a symbolic sample of dtype
		# int64 by default. If we want to keep our computations in floatX
		# for the GPU we need to specify to return the dtype floatX

		# here we could also do a for loop for each scaling value, but that'd be tedious.
		v1_sample = self.theano_rng.multinomial(
			n = self.scaling,
			pvals = v1_mean / 1.05,
			dtype = theano.config.floatX)

		return [pre_sigmoid_v1, v1_mean, v1_sample]

	def gibbs_hvh(self, h0_sample):
		''' This function implements one step of Gibbs sampling,
			starting from the hidden state'''
		pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
		return [pre_sigmoid_v1, v1_mean, v1_sample,
				pre_sigmoid_h1, h1_mean, h1_sample]

	def reconstruction_cost(self, v_sample, v_fantasy):
		"""
		reconstruction distance using L_2 (Euclidean distance) norm
		"""
		 # L_2 norm
		return (v_sample - v_fantasy).norm(2)

	def gibbs_vhv(self, v0_sample):
		''' This function implements one step of Gibbs sampling,
			starting from the visible state'''
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
		pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
		# theano assumes you only keep the last value.
		return [pre_sigmoid_h1, h1_mean, h1_sample,
				pre_sigmoid_v1, v1_mean, v1_sample]

	def get_cost_updates(self, lr=0.1, persistent=None, k=1):
		"""This functions implements one step of CD-k or PCD-k

		:param lr: learning rate used to train the RBM

		:param persistent: None for CD. For PCD, shared variable
			containing old state of Gibbs chain. This must be a shared
			variable of size (batch size, number of hidden units).

		:param k: number of Gibbs steps to do in CD-k/PCD-k

		Returns a proxy for the cost and the updates dictionary. The
		dictionary contains the update rules for weights and biases but
		also an update of the shared variable used to store the persistent
		chain, if one is used.

		"""
		# compute positive phase
		pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

		# decide how to initialize persistent chain:
		# for CD, we use the newly generate hidden sample
		# for PCD, we initialize from the old state of the chain
		if persistent is None:
			chain_start = ph_sample
		else:
			chain_start = persistent

		# perform actual negative phase
		# in order to implement CD-k/PCD-k we need to scan over the
		# function that implements one gibbs step k times.
		# Read Theano tutorial on scan for more information :
		# http://deeplearning.net/software/theano/library/scan.html
		# the scan will return the entire Gibbs chain
		[pre_sigmoid_nvs, nv_means, nv_samples,
		 pre_sigmoid_nhs, nh_means, nh_samples], updates = \
			theano.scan(self.gibbs_hvh,
					# the None are place holders, saying that
					# chain_start is the initial state corresponding to the
					# 6th output
					outputs_info=[None,  None,  None, None, None, chain_start],
					n_steps=k)

		# determine gradients on RBM parameters
		# not that we only need the sample at the end of the chain
		chain_end = nv_samples[-1]

		cost = T.mean(self.free_energy(self.input, self.scaling)) - T.mean(
			self.free_energy(chain_end, self.scaling))
		# We must not compute the gradient through the gibbs sampling
		gparams = T.grad(cost, self.params, consider_constant=[chain_end])

		# constructs the update dictionary
		for gparam, param in zip(gparams, self.params):
			# make sure that the learning rate is of the right dtype
			updates[param] = param - gparam * T.cast(lr,dtype=theano.config.floatX)

		if persistent:
			# Note that this works only if persistent is a shared variable
			updates[persistent] = nh_samples[-1]
			# pseudo-likelihood is a better proxy for PCD
			monitoring_cost = self.get_pseudo_likelihood_cost(updates)
		else:
			# reconstruction cross-entropy is a better proxy for CD
			monitoring_cost = self.reconstruction_cost(self.input, chain_end)

			#self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])

		return monitoring_cost, updates

	def get_pseudo_likelihood_cost(self, updates):
		"""Stochastic approximation to the pseudo-likelihood"""

		# index of bit i in expression p(x_i | x_{\i})
		bit_i_idx = theano.shared(value=0, name='bit_i_idx')

		# binarize the input image by rounding to nearest integer
		xi = T.round(self.input)

		# calculate free energy for the given bit configuration
		fe_xi = self.free_energy(xi, self.scaling)

		# flip bit x_i of matrix xi and preserve all other bits x_{\i}
		# Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
		# the result to xi_flip, instead of working in place on xi.
		xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - T.ceil(xi[:, bit_i_idx] / (xi[:, bit_i_idx] + 1)))

		# calculate free energy with bit flipped
		fe_xi_flip = self.free_energy(xi_flip, self.scaling)

		# equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
		cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

		# increment bit_i_idx % number as part of updates
		updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

		return cost

	def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
		"""Approximation to the reconstruction error

		Note that this function requires the pre-sigmoid activation as
		input.  To understand why this is so you need to understand a
		bit about how Theano works. Whenever you compile a Theano
		function, the computational graph that you pass as input gets
		optimized for speed and stability.  This is done by changing
		several parts of the subgraphs with others.  One such
		optimization expresses terms of the form log(sigmoid(x)) in
		terms of softplus.  We need this optimization for the
		cross-entropy since sigmoid of numbers larger than 30. (or
		even less then that) turn to 1. and numbers smaller than
		-30. turn to 0 which in terms will force theano to compute
		log(0) and therefore we will get either -inf or NaN as
		cost. If the value is expressed in terms of softplus we do not
		get this undesirable behaviour. This optimization usually
		works fine, but here we have a special case. The sigmoid is
		applied inside the scan op, while the log is
		outside. Therefore Theano will only see log(scan(..)) instead
		of log(sigmoid(..)) and will not apply the wanted
		optimization. We can not go and replace the sigmoid in scan
		with something else also, because this only needs to be done
		on the last step. Therefore the easiest and more efficient way
		is to get also the pre-sigmoid activation as an output of
		scan, and apply both the log and sigmoid outside scan such
		that Theano can catch and optimize the expression.

		"""

		# propdown is done using softmax, not sigmoid here !!
		# look at other rsm file for this.
		cross_entropy = T.mean(
				T.sum(self.input * T.log(T.nnet.softmax(pre_sigmoid_nv)) +
				(1 - self.input) * T.log(1 - T.nnet.softmax(pre_sigmoid_nv)),
					  axis=1))

		return cross_entropy


def test_rsm(training_set_size = 500, validation_set_size = 200, n_hidden = 200):
	from batch_data import BatchData as Batch
	import utils
	
	training_set_name   = "training_set_%d" % training_set_size
	validation_set_name = "validation_set_%d" % validation_set_size

	# We make sure Mongo is running somewhere :
	utils.connect_to_database(database_name = 'yelp')

	def load_dataset(size = 500, lexicon = None, name="training_set"):
		# if training_set.npy doesn't exist:
		rc = utils.ResourceConverter(lexicon = lexicon)
		batch = Batch(
		    data=utils.mongo_database_global['restaurants'].find({}, {'signature':1}), # from Mongo's cursor enumerator
		    batch_size = size,  # mini-batch
		    shuffle = True, # stochastic
		    conversion = rc.process # convert to matrices using lexicon)
		)
		dataset = batch.next()
		# and save it for later.
		numpy.save(name, dataset)
		return dataset

	if file_exists("lexicon.gzp"):
		lexicon = utils.Lexicon.load("lexicon.gzp")
	else:
		# if lexicon.gzp doesnt exist:
		# 'restaurants' is the name of the collection, we stem the words in the triggers,
		# and we lowercase them to minimize the visible dimensions (bag of words dimensions)
		lexicon = utils.gather_lexicon('restaurants',
		                               stem= True, 
		                               lowercase = True,
		                               show_progress= True)
		lexicon.save("lexicon.gzp")

	if file_exists("%s.npy" % training_set_name):
		train_set_x_mem      = numpy.load("%s.npy" % training_set_name)
	else:
		train_set_x_mem      = load_dataset(size = training_set_size, lexicon = lexicon, name=training_set_name)

	if file_exists("%s.npy" % validation_set_name):
		validation_set_x_mem = numpy.load("%s.npy" % validation_set_name)
	else:
		validation_set_x_mem = load_dataset(size = validation_set_size, lexicon = lexicon, name=validation_set_name)

	train_set_x      = theano.shared(train_set_x_mem,      borrow = True)
	validation_set_x = theano.shared(validation_set_x_mem, borrow = True)

	# construct the RSM class
	mini_batch_size = 100
	# allocate symbolic variables for the data
	n_train_batches = floor(train_set_x.get_value(borrow=True).shape[0] / mini_batch_size)
	rng             = numpy.random.RandomState(123)
	theano_rng      = T.shared_randomstreams.RandomStreams(rng.randint(2 ** 30))
	rsm             = RSM(n_visible=lexicon.max_index, n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

	def save_rsm(name=''):
		# save computation results:
		numpy.save(("%s_W_trained"     % name),     rsm.W.get_value(borrow=True))
		numpy.save(("%s_hbias_trained" % name), rsm.hbias.get_value(borrow=True))
		numpy.save(("%s_vbias_trained" % name), rsm.vbias.get_value(borrow=True))

	# get training function
	learning_rate  = theano.shared(0.01)
	cost, updates = rsm.get_cost_updates(lr=learning_rate, k=2)

	index = T.lscalar()    # index to a [mini]batch
	train_rsm = theano.function(
	    [index],
	    cost,
	    updates=updates,
	    givens={
	        rsm.input:   train_set_x[index * mini_batch_size:(index + 1) * mini_batch_size],
	        rsm.scaling: train_set_x[index * mini_batch_size:(index + 1) * mini_batch_size].sum(axis=1).astype(theano.config.floatX),
	    },
	    name='train_rbm')


	[pre_sigmoid_h1, h1_mean, h1_sample,
				pre_sigmoid_v1, v1_mean, v1_sample] = rsm.gibbs_vhv(rsm.input)

	validate_rsm = theano.function(
		[],
		rsm.reconstruction_cost(rsm.input, v1_sample),
		givens = {
			rsm.input: validation_set_x,
			rsm.scaling: validation_set_x.sum(axis=1)
		}
	)


	training_epochs     = 300 # will stop early
	batch_indices = [i for i in range(n_train_batches)]
	start_time = time.time()
	min_val = None
	try:
		for epoch in range(training_epochs):

			# go through the training set
			mean_cost = []

			# more stochasticity:
			random.shuffle(batch_indices)

			for batch_index in batch_indices:
			    mean_cost.append(train_rsm(batch_index) / mini_batch_size)

			validation_cost = validate_rsm()           / validation_set_size

			print('Training epoch %d, cost is %.4f, validation cost is %.4f' % (epoch+1, numpy.mean(mean_cost), validation_cost))
		
			if min_val != None and validation_cost < min_val:
				save_rsm("min_validation")
			min_val = min(min_val, validation_cost) if min_val != None else validation_cost

		print('Training took %.05fmn' % ((time.time() - start_time)/60.0))
	except (KeyboardInterrupt, SystemExit):
		print("Saving final rsm...")
		save_rsm("final")
		exit()
	except:
		raise
	print("Saving final rsm...")
	save_rsm("final")

if __name__ == '__main__':
	test_rsm(training_set_size = 3000, validation_set_size = 600)