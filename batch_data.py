import random

class BatchData(object):
	"""
	Interface with cursor in Mongo to perform Stochastic Gradient Descent.
	"""
	def __init__(self, data=None, batch_size=100, auto_rewind = False, shuffle=False, conversion=None):
		"""
		Initialize a batch processor w/. Mongo for Stochastic Gradient Descent.
		"""
		assert(data != None),              "Batch data cannot be None."
		assert(hasattr(data, '__iter__')), "Batch data must be an enumerable."
		assert(hasattr(data, 'count')),    "Batch data must have a total count."
		self._data = data
		self.auto_rewind = auto_rewind
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.conversion = conversion
		self.index = 0

	# https://docs.python.org/2/tutorial/classes.html#iterators
	# Following their example:
	def __iter__(self):
		return self

	class NoMoreData(Exception):
		def __init__(self, message=''):
			self.message = message

	def to_shared_variable(self):
		import theano
		return theano.shared(self.next(), borrow = True)

	def create_batch(self):
		return [i for i in self._data[self.index:self.index+self.batch_size]]

	def reset_cursor(self):
		self._data.rewind()

	def get_raw_batch(self):
		# reset the cursor to perform additional requests to Mongo
		self.reset_cursor()
		return self.create_batch()

	def random_start_point(self):
		return random.randint(0, self._data.count()-self.batch_size)

	def auto_rewind_index(self):
		if self.shuffle:
			self.index = self.random_start_point()
		elif self._data.count() < self.index:
			self.index = 0

	def next(self):
		if self.auto_rewind or self.shuffle:
			self.auto_rewind_index()
		batch = self.get_raw_batch()
		if self.conversion:
			batch = self.conversion(batch)
		if len(batch) == 0 and not self.auto_rewind:
			raise BatchData.NoMoreData("No More Data Left. Set auto_rewind or shuffle to True to avoid these troubling messages.")
		self.index = self.index + self.batch_size
		return batch