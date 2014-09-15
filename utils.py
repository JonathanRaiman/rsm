import numpy as np
from pymongo import MongoClient
from matplotlib import pyplot as plt
from stemming.porter2 import stem
import uuid
import gzip
import pickle
from math import ceil
from IPython.display import HTML, Javascript, display

mongo_database_global = None
mongo_client_global   = None

def progress_bar():
	"""
	Create a progress bar in Ipython and show progress.

	usage:
	[div_id, updater] = utils.progress_bar()

	for i in range(1,101):
    	time.sleep(0.1)
    	updater(i)
    """
	divid = str(uuid.uuid4())

	pb = HTML(
	"""
	<div class="progress">
	  <div id="%s" class="bar" role="progressbar" aria-valuenow="60" aria-valuemin="0" aria-valuemax="100" style="width: 0%%;">
	    0%%
	  </div>
	</div>
	""" % divid)
	display(pb)
    
    # we create an updater method that gets returned
    # so user can manually re-update the progress bar.
	def update_progress_bar(i):
		display(Javascript("""
var el = document.getElementById('%s');
el.style.width = '%i%%';
el.innerHTML   = '%i%%';
""" % (divid, i, i)))

	return (divid, update_progress_bar)

def convert_row_to_word_representation(lexicon, row):
	return [lexicon.decode(i) for i, word in enumerate(row) if word > 0]

def to_word_representation(lexicon, matrix):
	"""
	Convert a matrix of word reps using a lexicon
	Inputs:
		reverse_lexicon: words ordered by their id
		matrix: matrix with column position corresponding to word id
	Outputs:
		matrix of words.
	"""
	converted_matrix = np.empty(matrix.shape[0], dtype='object_')

	for index, row in enumerate(matrix):
		converted_matrix[index] = convert_row_to_word_representation(lexicon, row)

	return converted_matrix

def get_mongo_client(hostname = 'localhost', port = 27017):
	global mongo_client_global
	mongo_client_global = MongoClient(hostname, port)

def connect_to_database(database_name='test'):
	global mongo_client_global
	global mongo_database_global
	if not mongo_client_global:
		get_mongo_client()
	mongo_database_global = mongo_client_global[database_name]

class NotConnectedError(Exception):
	def __init__(self, message):
		self.message = message

class ResourceLabeler(object):
	def __init__(self, lexicon, accessor, converter=None):
		self.lexicon = lexicon
		self.accessor = accessor
		self.converter = converter

	@staticmethod
	def convert_mongo_resources_to_numpy_label(resources, lexicon, accessor, converter=None):
		labels   = np.ones(len(resources), dtype='int32') * len(lexicon.items())
		for index, post in enumerate(resources):
			if type(post.get(accessor)) is not list:
				if converter is not None:
					key = lexicon.get(converter(post.get(accessor)))
					if key != None:
						labels[index] = key
				else:
					key = lexicon.get(post.get(accessor))
					if key != None:
						labels[index] = key
			else:
				for label in post.get(accessor):
					if converter is not None:
						key = lexicon.get(converter(label))
						if key != None:
							labels[index] = key
							break
					else:
						key = lexicon.get(label)
						if key != None:
							labels[index] = key
							break

		return labels

	def process(self, resources):
		return self.convert_mongo_resources_to_numpy_label(resources, self.lexicon, self.accessor, self.converter)

class ResourceConverter(object):
	def __init__(self, lexicon):
		self.lexicon = lexicon

	@staticmethod
	def convert_mongo_resources_to_numpy(resources, lexicon, dtype='float32'):
		learning_data   = np.zeros(
			[
				len(resources),
				len(lexicon.keys())
			],
		dtype=dtype)
		for index, post in enumerate(resources):
			for concept in post.get('signature'):
				key = lexicon.get_with_normalization(concept.get('trigger'))
				if key != None:
					learning_data[index, key] = concept.get('df')
		return learning_data

	def process(self, resources, dtype = 'float32'):
		return self.convert_mongo_resources_to_numpy(resources, self.lexicon, dtype = dtype)

def import_from_mongo(collection_name, stopping_point=1000, lexicon = None, dtype = 'float32', **kwargs):
	global mongo_database_global
	if not mongo_database_global:
		raise NotConnectedError("Database Not initialized. Call connect_to_database.")

	collection      = mongo_database_global[collection_name]
	lexicon = lexicon if (lexicon != None) else gather_lexicon(collection, **kwargs)
	dictionary_size = len(lexicon.keys())

	learning_data   = np.zeros(
		[
			min(stopping_point, collection.count()),
			dictionary_size
		],
	dtype=dtype)
	for index, post in enumerate(collection.find({})[0:min(stopping_point, collection.count())]):
		for concept in post.get('signature'):
			key = lexicon.get_with_normalization(concept.get('trigger'))
			if key:
				learning_data[index, key] = concept.get('df')
	return (learning_data, lexicon)

def gather_stats(collection_name):
	if not mongo_database_global:
		raise NotConnectedError("Database Not initialized. Call connect_to_database.")
	counts    = {}
	collection = mongo_database_global[collection_name]
	for post in collection.find({}):
		for concept in post.get('signature'):
			if not counts.get(concept.get('trigger')):
				counts[concept.get('trigger')] = concept.get('df')
			else:
				counts[concept.get('trigger')] += concept.get('df')
	return counts

def plot_progress(progress_scores, xlabel='Training Epoch', ylabel='Progress'):
	fig = plt.figure()
	score_length = len(progress_scores)
	axes = fig.add_axes([0.1, 0.1, 1.0, 0.8])
	axes.plot(np.linspace(1, score_length, score_length), progress_scores, 'r')
	axes.set_xlabel(xlabel)
	axes.set_ylabel(ylabel);


def create_lexicon_from_strings(strings, lowercase = False, stem = False):
	lexicon = Lexicon(lowercase=lowercase, stem=stem)
	for string in strings:
		lexicon.set_if_empty_normalization(string)
	return lexicon

class Lexicon(dict):
	def __init__(self, *args, **kwargs):
		self.lowercase       = True if (kwargs.get('lowercase') == True) else False
		self.stem            = True if (kwargs.get('stem')      == True) else False
		self.reverse_lexicon = []
		self.max_index       = 0
		super().__init__(*args)

	def normalize(self,key):
		if self.lowercase:
			return stem(key.lower()) if self.stem else key.lower()
		else:
			return stem(key.lower()) if self.stem else key.lower()

	def get_with_normalization(self, key):
		return self.get(self.normalize(key))

	def decode(self, key):
		# does not check if key is past lexicon size
		return self.reverse_lexicon[key]

	def set_if_empty_normalization(self, key):
		"""
		Makes an assignment using the Lexicon normalizations
		(stemming and lowercasing) if the key is empty (None).
		Inputs:
			key:
			value:
		Outputs:
			False if not assigment is made,
			Normalized key if assignment is made.
		"""
		norm_key = self.normalize(key)
		if self.get(norm_key) == None:
			self[norm_key] = self.max_index
			self.reverse_lexicon.append(norm_key)
			self.max_index += 1
			return norm_key
		else:
			return False

	def save(self, path):
		with gzip.open(path, 'wb') as file:
			pickle.dump(self, file, 1)

	@staticmethod
	def load(path):
		file = gzip.open(path, 'r')
		saved_lexicon = pickle.load(file)
		file.close()
		return saved_lexicon

def gather_lexicon(collection_name, lowercase = False, stem = False, show_progress = False):
	if not mongo_database_global:
		raise NotConnectedError("Database Not initialized. Call connect_to_database.")
	lexicon    = Lexicon(lowercase=lowercase, stem=stem)
	collection = mongo_database_global[collection_name]
	total      = collection.count()
	[div_id, updater] = progress_bar() if show_progress else (None, None)

	for index, post in enumerate(collection.find({}, {'signature': 1})):
		for concept in post.get('signature'):
			lexicon.set_if_empty_normalization(concept.get('trigger'))

		if show_progress:
			updater( ceil( 100 * index / total) )

	return lexicon