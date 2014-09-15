from rsm import RSM
import theano as T
import numpy as np

class RSM_SVRG(RSM):

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
		self.mu_weight_matrix = T.shared(
			np.zeros(self.weight_matrix_size, dtype=T.config.floatX),
			'weight_matrix',
			strict = False,
			borrow = True
			)
		self.num_steps = T.shared(np.int32(0))
		self.mu_hidden_bias = T.shared(
			np.zeros(self.number_of_hidden_units, dtype=T.config.floatX),
			'weight_matrix',
			strict = False,
			borrow = True
			)
		self.mu_visible_bias = T.shared(
			np.zeros(self.number_of_visible_units, dtype=T.config.floatX),
			'weight_matrix',
			strict = False,
			borrow = True
			)

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
		weight_matrix_update    = T.tensor.dot(visible_start.T, hidden_start) - T.tensor.dot(visible_fantasy.T, hidden_fantasy)
		weight_matrix_grad_step = T.tensor.cast(weight_matrix_update - self.old_step_weight_matrix + self.mu_weight_matrix / T.tensor.maximum(1, self.num_steps), dtype=T.config.floatX)
		
		visible_bias_update    = visible_start.sum(axis=0) - visible_fantasy.sum(axis=0)
		visible_bias_grad_step = T.tensor.cast(visible_bias_update - self.old_step_visible_bias + self.mu_visible_bias / T.tensor.maximum(1, self.num_steps), dtype=T.config.floatX)

		hidden_bias_update     = hidden_start.sum(axis=0) - hidden_fantasy.sum(axis=0)
		hidden_bias_grad_step  = T.tensor.cast(hidden_bias_update - self.old_step_hidden_bias + self.mu_hidden_bias / T.tensor.maximum(1, self.num_steps), dtype=T.config.floatX)

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
					self.weight_matrix + T.tensor.cast(self.learning_rate, dtype=T.config.floatX) * weight_matrix_grad_step
				),
				(
					self.old_step_hidden_bias,
					hidden_bias_update
				),
				(
					self.hidden_bias,
					self.hidden_bias + T.tensor.cast(self.learning_rate, dtype=T.config.floatX) * hidden_bias_grad_step
				),
				(
					self.old_step_visible_bias,
					visible_bias_update
				),
				(
					self.visible_bias,
					self.visible_bias + T.tensor.cast(self.learning_rate, dtype=T.config.floatX) * visible_bias_grad_step
				),
				(
					self.mu_hidden_bias,
					self.mu_hidden_bias + hidden_bias_update
				),
				(
					self.mu_visible_bias,
					self.mu_visible_bias + visible_bias_update
				),
				(
					self.mu_weight_matrix,
					self.mu_weight_matrix + weight_matrix_update
				),
				(
					self.num_steps,
					self.num_steps + 1
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