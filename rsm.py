# coding: utf-8
import numpy as np
import theano as T
import pickle, gzip

# Install developement versions because the rest have non-fantastic support for Ipython3:
# pip3 install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# pip3 install --upgrade git+https://github.com/matplotlib/matplotlib.git#egg=matplotlib-dev



class RSM(object):
	"""
	**Replicated Softmax Machine:**

		Implementation of the Replicated Softmax model,
		as presented by R. Salakhutdinov & G.E. Hinton
		in http://www.mit.edu/~rsalakhu/papers/repsoft.pdf

	"""

	def check_assertions(self):
		if self.data != None:
			assert(self.data.shape[1] == self.number_of_visible_units), "Data row size must equal number of visible units"

	def create_rng(self, rng=None):
		self.rng = T.tensor.shared_randomstreams.RandomStreams(seed=123)
		if rng and isinstance(rng, T.tensor.shared_randomstreams.RandomStreams):
			self.rng.gen_seedgen = rng.gen_seedgen

	@property
	def name(self):
		return "RSM"

	@property
	def weight_matrix_size(self):
		"""
		The dimensions of the weights matrix for this RSM, with
		# visible units ⨉ # hidden units:
		Outputs:
			[# visible units, # hidden units]
		"""
		return [self.number_of_visible_units, self.number_of_hidden_units]

	@weight_matrix_size.setter
	def weight_matrix_size(self,value):
		assert(len(value) == 2), "Specify size as a length 2 array [# visible units, # hidden units]"
		assert(value[0] > 0),    "weight matrix needs positive dimensions"
		assert(value[1] > 0),    "weight matrix needs positive dimensions"
		self.number_of_visible_units = value[0]
		self.number_of_hidden_units  = value[1]
		self.weight_matrix.resize(self.weight_matrix_size)

	def create_shared_variables(self):
		"""
		Creates the theano Shared variables for:
		visible bias, hidden bias, weight matrix, and
		previous steps for each to enable momentum in CD-k.
		"""
		self.old_step_weight_matrix = T.shared(
			np.zeros(self.weight_matrix_size, dtype=T.config.floatX),
			'old_step_weight_matrix',
			strict = False,
			borrow = True
			)
		self.weight_matrix = T.shared(
			(self.weight_initialization * np.random.randn(
				self.number_of_visible_units,
				self.number_of_hidden_units)).astype(T.config.floatX),
			'weight_matrix',
			strict = False,
			borrow = True
			)
		self.hidden_bias = T.shared(
			np.zeros(self.number_of_hidden_units, dtype=T.config.floatX),
			'weight_matrix',
			strict = False,
			borrow = True
			)
		self.visible_bias = T.shared(
			np.zeros(self.number_of_visible_units, dtype=T.config.floatX),
			# (self.weight_initialization * np.random.randn(self.number_of_visible_units)
			# ).astype(T.config.floatX),
			'weight_matrix',
			strict = False,
			borrow = True
			)
		self.old_step_hidden_bias = T.shared(
			np.zeros(self.number_of_hidden_units, dtype=T.config.floatX),
			'weight_matrix',
			strict = False,
			borrow = True
			)
		self.old_step_visible_bias = T.shared(
			np.zeros(self.number_of_visible_units, dtype=T.config.floatX),
			'weight_matrix',
			strict = False,
			borrow = True
			)

	def project_into_hidden_layer(self, visible):
		"""
		Computes a projection into the hidden layer
		Inputs:
			@visible: the data to project
		Outputs:
			projection: the data in hidden unit space.
		"""
		scaling           = visible.sum(axis=1).astype(T.config.floatX)
		hidden_projection = self.propup(visible, scaling)
		return (hidden_projection, scaling)

	def create_weights(self):
		"""
		Create Theano shared variables for RSM,
		create sampling functions, and update rules
		using Theano functins & updates.
		"""
		# create Theano shared variables:
		self.create_shared_variables()

		# Symbolically define the parameters:
		visible_start   = T.tensor.fmatrix('visible_start')
		hidden_start    = T.tensor.fmatrix('hidden_start')
		visible_fantasy = T.tensor.fmatrix('visible_fantasy')
		hidden_fantasy  = T.tensor.fmatrix('hidden_fantasy')
		scaling         = T.tensor.fvector('scaling')

		# define symbolic update steps:
		weight_matrix_update = (
			self.old_step_weight_matrix * self.momentum + 
			T.tensor.dot(visible_start.T, hidden_start) -
			T.tensor.dot(visible_fantasy.T, hidden_fantasy))
		

		visible_bias_update = T.tensor.cast(
				(
					self.old_step_visible_bias * self.momentum + 
					visible_start.sum(axis=0) - visible_fantasy.sum(axis=0)
				), dtype=T.config.floatX)

		hidden_bias_update = (
			self.old_step_hidden_bias * self.momentum + 
			hidden_start.sum(axis=0) - hidden_fantasy.sum(axis=0))

		# create sigmoid function using the shared variables:
		self.propup = T.function(
			[visible_start, scaling],
			T.tensor.nnet.sigmoid(
				T.tensor.dot(visible_start, self.weight_matrix) + 
				T.tensor.outer(scaling, self.hidden_bias)
			)
		)

		# create propdown with likelihood cross-entropy metric:
		desired_conjoined_output = T.tensor.dot(
			hidden_start,
			self.weight_matrix.T
		) + self.visible_bias

		propdown_observation = T.tensor.nnet.softmax(desired_conjoined_output);

		# deprecated costs was:
		#T.tensor.nnet.categorical_crossentropy(propdown_observation, visible_start).sum()

		# nansum would be better here:

		propdown_cost = (visible_start * T.tensor.log(propdown_observation)).sum()

		self.propdown = T.function(
			[
				visible_start,
				hidden_start
			],
			[
				propdown_observation,
				propdown_cost
			]
		)

		# mean squared error cost:
		self.get_mse_cost = T.function([visible_start,visible_fantasy], (visible_start - visible_fantasy).norm(L=2))

		# Create symbolic function for update of parameters:
		self.update_weights = T.function(
			[
				visible_start,
				hidden_start,
				visible_fantasy,
				hidden_fantasy
			],
			self.weight_matrix,
			updates= [
				(
					self.old_step_weight_matrix,
					weight_matrix_update
				),
				(
					self.weight_matrix,
					self.weight_matrix + T.tensor.cast(self.learning_rate, dtype=T.config.floatX) * weight_matrix_update
				),
				(
					self.old_step_hidden_bias,
					hidden_bias_update
				),
				(
					self.hidden_bias,
					self.hidden_bias + T.tensor.cast(self.learning_rate, dtype=T.config.floatX) * hidden_bias_update
				),
				(
					self.old_step_visible_bias,
					visible_bias_update
				),
				(
					self.visible_bias,
					self.visible_bias + T.tensor.cast(self.learning_rate, dtype=T.config.floatX) * visible_bias_update
				)
			]
		)

		# Create multinomial sampling function (from GPU):
		num   = T.tensor.fscalar('num')
		probs = T.tensor.fvector('probs')

		# for numerical stability we divide by 1.05
		self.sample_visible_from_hidden = T.function(
			[num, probs],
			self.rng.multinomial( n= num, pvals = probs / 1.05, dtype = T.config.floatX), allow_input_downcast=True)

		# Create binomial sampling function (from GPU):
		self.sample_hidden_from_visible = T.function(
			[visible_start, scaling],
			self.rng.binomial(
				n=1,
				p=T.tensor.nnet.sigmoid(
					T.tensor.dot(visible_start, self.weight_matrix) + 
					T.tensor.outer(scaling, self.hidden_bias)
				),
				dtype=T.config.floatX
			)
		)

		visible_probs   = T.tensor.fmatrix('visible_probs')
		self.sample_hidden_from_visible_probs = T.function(
			[visible_probs],
			self.rng.binomial(
				n=1,
				p=visible_probs,
				dtype=T.config.floatX
			)
		)

	def __init__(self,
		momentum = 0.9,
		k = 1,
		data = None,
		learning_rate = 0.001,
		hidden_units = 100,
		visible_units = None,
		weight_initialization = 0.001,
		use_random_hidden_sampling = False,
		rng = None ):

		"""
		Initialize a CD-k trainer with momentum and k Gibbs steps.
		
		@param hidden_units: Number of hidden units (latent topic models)
		@param momentum: Momentum to use for contrastive divergence
		@param data: training data, row-size.
		@param rng: a pre-existing random number generator to boostrap
					this Gibbs chain.
		@param k: number of gibbs steps k in training
		@param use_random_hidden_sampling: sampling from hidden during CD-k
										   is initialized using binomials
										   sampling. Off by default.

		"""

		self.momentum              = momentum
		self.k                     = k
		self._data                 = data
		self.number_of_words       = np.sum(data)
		# store the learning rate with the rest of the shared variables,
		# so that updates are reflected in calculations:
		self._learning_rate        = T.shared(
			np.float64(learning_rate),
			strict = False)
		self.weight_initialization = weight_initialization

		self.number_of_visible_units = data.shape[1] if data != None else (visible_units if visible_units != None else None)
		self.number_of_hidden_units  = hidden_units

		self.use_random_hidden_sampling = use_random_hidden_sampling

		self.check_assertions()

		self.create_rng(rng);
		self.create_weights();

	@property
	def learning_rate(self):
		return self._learning_rate

	@learning_rate.setter
	def learning_rate(self, value):
		self._learning_rate.set_value(value)

	@property
	def data(self):
		return self._data

	@data.setter
	def data(self,value):
		assert(value != None), "Data cannot be empty."
		assert(value.shape[1] == self.number_of_visible_units), "Data row size must equal number of visible units"
		self._data            = value
		self.number_of_words  = np.sum(self._data)
		
	def sample_visible(self, probabilities, scaling):
		"""
		Samples a multinomial distribution using the matrix of probabilities provided
		outputs:
			activations for visible layer.
		"""
		visible = np.zeros_like(probabilities)
		for i in range(probabilities.shape[0]):
			visible[i] = self.sample_visible_from_hidden(
				scaling[i],
				probabilities[i])
		return visible

	def contrastive_divergence(self,
		visible = None,
		hidden  = None,
		scaling = None):
		"""
		Performs a Gibbs sampling step of contrastive divergence using visible
		and hidden units.

		Inputs:
			@param visible: the visible activation
			@param hidden: the hidden activation
			@param scaling: the replicated softmax scaling parameter
				(the size of a document)
		Output:
			(visible_fantasy, hidden_fantasy, likelihood)
		"""

		[visible_fantasy_pdf, likelihood] = self.propdown(
			visible,
			hidden)

		visible_fantasy = self.sample_visible(
			visible_fantasy_pdf,
			scaling)

		hidden_fantasy = self.propup(
			visible_fantasy,
			scaling)

		return (
			visible_fantasy,
			hidden_fantasy,
			likelihood
			)
	def reconstruct(self, visible):
		"""
		Attempts to reconstruct the input using hidden activations.
		Inputs:
			visible: the visible units, row-wise observations
		Outputs:
			a numpy array of hidden activations,
			and the likelihood of the propagation.
		"""
		scaling_factor = visible.sum(axis=1).astype(T.config.floatX)

		return self.sample_visible(self.propdown(
			visible,
			self.sample_hidden(visible, scaling_factor)
			)[0], scaling_factor)

	# def sample_hidden_from_visible(self, visible):
	# 	"""
	# 	Propagates the visible layer into the hidden layer.
	# 	Inputs:
	# 		visible: the visible units, row-wise observations
	# 	Outputs:
	# 		a numpy array of hidden activations.
	# 	"""
	# 	scaling_factor = visible.sum(axis=1).astype(T.config.floatX)
	# 	return self.sample_hidden(visible, scaling_factor)

	def sample_hidden(self, visible, scaling):
		"""
		Propagates the visible layer into the hidden layer.

		Inputs:
			visible: the visible units, row-wise observations
			scaling: the column-size sum of each observation for normalization.
		Outputs:
			a numpy array of hidden activations.
		"""
		return self.sample_hidden_from_visible(visible, scaling)

	def train(self):
		"""
		Perform one step of steepest descent on parameters

		Inputs:
			None
		Outputs:
			perplexity (float)
		"""
		# the input data:
		visible_start = self.data

		# hidden biases scaling factor:
		(hidden_start, scaling_factor) = self.project_into_hidden_layer(
			visible = visible_start)
		#hidden_start_rand = self.sample_hidden_from_visible_probs(hidden_start) if self.use_random_hidden_sampling else hidden_start

		visible_fantasy = visible_start
		hidden_fantasy  = hidden_start

		# obtain the likelihood of the step:
		likelihood = 0.0

		# Perform Gibbs sampling steps:
		for step in range(self.k):
			# here we use hidden_start_rand instead of hidden_start
			# because we use the sampled value
			(visible_fantasy, hidden_fantasy, step_likelihood) = self.contrastive_divergence(
				visible_fantasy,
				hidden_fantasy,
				scaling_factor
				)
			if step == 0:
				likelihood += step_likelihood

		# Update weights based on fantasies:
		# for training we use hidden_start, not hidden_start_rand, to use
		# the expected value
		self.update_weights(
			visible_start,
			hidden_start,
			visible_fantasy,
			hidden_fantasy
			)

		# perplexity:
		return np.exp(- likelihood / self.number_of_words)

		# # mean squared error:
		# return np.mean(self.get_mse_cost(visible_fantasy, visible_start) / np.float64(self.number_of_visible_units * visible_start.shape[0]))
	def to_hash(self):
		return {
				'weight_matrix': self.weight_matrix.get_value(),
				'hidden_bias': self.hidden_bias.get_value(),
				'visible_bias': self.visible_bias.get_value(),
				'momentum': self.momentum,
				'k': self.k,
				'rng': self.rng,
				'learning_rate': self.learning_rate.get_value()
			}
			
	def save(self, path):
		with gzip.open(path, 'wb') as file:
			pickle.dump(self.to_hash(), file, 1)

	@staticmethod
	def load(path):
		file = gzip.open(path, 'r')
		saved_rsm = pickle.load(file)
		file.close()
		new_rsm = RSM(
			visible_units = saved_rsm['weight_matrix'].shape[0],
			hidden_units  = saved_rsm['weight_matrix'].shape[1],
			k = saved_rsm['k'],
			momentum = saved_rsm['momentum'],
			learning_rate = saved_rsm['learning_rate'],
			rng = saved_rsm['rng'])
		new_rsm.visible_bias.set_value(saved_rsm['visible_bias'])
		new_rsm.hidden_bias.set_value(saved_rsm['hidden_bias'])
		new_rsm.weight_matrix.set_value(saved_rsm['weight_matrix'])
		return new_rsm

def test_rsm(momentum=0.9, hidden_units=123, epochs=200):
	"""
	Test the Replicated Softmax Machine.
	Finaly perplexity w/. 200 epochs should be near 280 or 360.

	Inputs:
		momentum: how much each previous weight update
			      impacts future updates (“ball rolling
			      down the gradient hill” ~ Hinton).
		hidden_units: number of latent topics
		epochs: number of training steps
	Outputs:
		None

	"""
	from daichi_rsm import fmatrix as importer
	datum   = importer.parse("sample_data/train")
	encoder = RSM(momentum=momentum, data=datum, hidden_units=hidden_units)
	errors  = np.zeros(epochs)
	for epoch in range(epochs):
		# update rule for gibbs steps (mixing rate proxy):
		#if epoch > 0 and epoch % 50 == 0:
		#	encoder.k = int(1 + (float(epoch) / float(epochs)) * 4)
		errors[epoch] = encoder.train()
		print("Epoch[%2d] : MSE = %.02f [iter=%d]" % (epoch, errors[epoch],encoder.k))
	encoder.train()

if __name__ == "__main__":
	test_rsm()