import theano, theano.tensor as T, numpy as np, scipy.io as scipy_io, pyximport, threading, logging, time, sys, os, pickle, gzip, json
from collections import OrderedDict
sys.path.append("/Users/jonathanraiman/Desktop/Coding/language_modeling/")
pyximport.install(setup_args={"include_dirs": np.get_include()})
from .train_model import train_sentence_concatenation, predict_sentence_window, predict_distribution_sentence_window
from gensim import utils as gensim_utils
from queue import Queue
from numpy import float32 as REAL, int32 as INT
logger = logging.getLogger("objectlm.training")

def bilinear_form(tensor, x):
    """
    Bilinear form like:
    x • A • x^T
    """
    return T.dot(T.dot(tensor, x), x)

class ObjectLM(object):
    """
    
    Object Language Model
    -------------------
    
    Use all sorts of information as a labeled dataset.
    
    Train using language model w/. words
    ------------------------------------
    
    e.g. use an indexing scheme as follows:
    
        'hello' -> 0
        'world' -> 1
        '.'     -> 2
        
    Count the number of words in vocab:
    
        model = ObjectLM(vocabulary_size = 3)
        
    And then train using those words:
    
        model.update_fun(
            [1, 2, 3],
            <index of object>,
            [<labels for multiclass>],
            [<labels for sigmoid>]
        )
    
    Train using indexed objects
    ---------------------------
    
    e.g. use an indexing scheme as follows:
    
        'hello world .' -> 0
        'hello Joe .'   -> 1
        
    Count the number of objects in the object vocabulary:
        
        model = ObjectLM(object_vocabulary_size = 2,
                       vocabulary_size= 1000)
                       
    And then train using those objects:
    
        model.update_fun(
            [<indices of words>],
            0,
            [<labels for multiclass>],
            [<labels for sigmoid>]
        )
    
    
    Train using sigmoid classes
    ---------------------------
    
    e.g. using 6 classes, with classes 3 and 5 active, others off:
    
        model.update_fun(
            [<indices of words>],
            <index of object>,
            [<labels for multiclass>],
            [0, 0, 1, 0, 1, 0])
        
    Train using multiclass labels
    -----------------------------
    
    e.g. using 2 multiclass labels, with 1st label being
    class 3, and 2nd label being class 4:
            
        model.update_fun(
            [<indices of words>],
            <index of object>,
            [2, 3],
            [<labels for sigmoid>]
        
    
    
    """

    version_id = 1.0
    
    def load_saved_weights(self, path):
        """
        Replace current model parameters by those
        present in a directory with the same names.

        Inputs
        ------

        str path: the path to the directory the parameters are stored at.

        """
        if not path.endswith("/"): path = path + "/"

        for param in self.params:
            mat = np.load(path  + param.name + ".npy")
            param.set_value(mat)

    def reset_weights(self):
        # in the typical case we consider a linear projection of the concatenation
        # or addition of inputs:
        if self.concatenate:
            projection_size = self.size * self.window + self.object_size
        else:
            assert(self.size == self.object_size), "In additive (not concatenated) projection, "
            "the hidden objects must have the same dimensionality as the language model vectors."
            projection_size = self.size

        proj_matrix = ((1.0 / self.size) * \
            np.random.randn(
                self.output_size,
                projection_size).astype(REAL))
        self.projection_matrix.set_value(proj_matrix)
        
        # in the case we want to use 2nd order behavior for modeling we add a tensor:
        if self.bilinear_form:
            # reset bilinear form matrix:
            random_comp_matrix = ((1.0 / (self.size)) * np.random.randn(
                self.output_size,
                projection_size,
                projection_size)).astype(REAL)
            self.composition_matrix.set_value(random_comp_matrix)
        
        
        # reset bias vector:
        bias_vector      = np.zeros(self.output_size, dtype=REAL)
        self.bias_vector.set_value(bias_vector)


        # reset language models:
        model_matrix = ((1.0 / self.size) * \
                        np.random.randn(self.vocabulary_size, self.size)).astype(REAL)
        self.model_matrix.set_value(model_matrix)

        object_matrix = ((1.0 / self.object_size) * \
                        np.random.randn(self.object_vocabulary_size, self.object_size)).astype(REAL)
        self.object_matrix.set_value(object_matrix)
            
    def save_model(self, path):
        """

        Save the parameters of this model to a directory.

        Inputs
        ------

        str path: the path to the directory the parameters will be stored.


        Notes
        -----
        Does not save other parameters and options yet (TODO!)

        """
        if not path.endswith("/"): path = path + "/"

        if not os.path.exists(path): os.makedirs(path)
        for param in self.params:
            np.save("%s%s.npy" % (path, param.name), param.get_value(borrow = True))

        self.save_model_parameters(path)

    def save_model_parameters(self, path):
        if not path.endswith("/"): path = path + "/"
        if not os.path.exists(path): os.makedirs(path)

        parameters_hash = {
            "version_id": type(self).version_id,
            "window": self.window,
            "size": self.size,
            "object_size": self.object_size,
            "output_classes" : list(self.output_classes),
            "output_sigmoid_classes": self.output_sigmoid_classes,
            "theano_mode" : self.theano_mode,
            "object_vocabulary_size": self.object_vocabulary_size,
            "batch_size": self._batch_size,
            "bilinear_form" : self.bilinear_form,
            "alpha" : self._alpha,
            "UnknownWordIndex" : self.vocab.UnknownWordIndex,
            "UnknownUppercaseWordIndex": self.vocab.UnknownUppercaseWordIndex
        }

        for k, c in enumerate(self.output_labels):
            parameters_hash["softmax_labels_%d" % (k)] = c

        parameters_hash["sigmoid_labels"] = self.output_sigmoid_labels


        with open(path + "__dict__.txt", "w") as f:
            for key, value in parameters_hash.items():
                f.write(key)
                f.write(" ")
                f.write(str(value))
                f.write("\n")

    def save_vocabulary(self, path):
        if not path.endswith("/"): path = path + "/"

        if not os.path.exists(path): os.makedirs(path)
        with gzip.open(path + "__vocab__.gz", "wt") as f:
            for word in self.vocab.index2word:
                f.write( (word + "\n") )

    def save_model_to_java(self, path):
        """

        Save the parameters of this model to a directory.

        Inputs
        ------

        str path: the path to the directory the parameters will be stored.

        """
        if not path.endswith("/"): path = path + "/"

        if not os.path.exists(path): os.makedirs(path)
        param_dict = {}

        # if we need to create a normalized version of the word and object vectors
        # we create one now.

        for param in self.params:
            param_dict[param.name] = param.get_value(borrow = True)

        scipy_io.savemat(path + "parameters.mat", param_dict)
        self.save_model_parameters(path)

    @property
    def batch_size(self):
        return self._batch_size
    
    def __init__(self,
                 vocabulary,
                 object_vocabulary_size = 100,
                 size = 20,
                 window = 10,
                 object_size = 20,
                 output_classes = [2, 2], # 2 output classes of size 2 each
                 output_labels = [],
                 output_sigmoid_classes = 4,
                 output_sigmoid_labels = [],
                 concatenate = True,
                 alpha = 0.035,
                 bilinear_form = False, # not support by cython yet.
                 theano_mode = 'FAST_RUN', # or FAST_COMPILE
                 batch_size = 100):
        
        self.alpha = theano.shared(np.float64(alpha).astype(REAL), name = 'alpha')
        self._alpha = float(alpha)
        self.output_classes = np.array(output_classes, dtype=INT)
        self.output_sigmoid_classes = output_sigmoid_classes
        self._batch_size = batch_size
        self.concatenate = concatenate
        self.bilinear_form = bilinear_form
        self.object_size = object_size
        self.size = size
        self.theano_mode = theano_mode
        self.vocab = vocabulary
        self.vocabulary_size = len(self.vocab.vocab)
        self.object_vocabulary_size = object_vocabulary_size
        self.window = window

        self.output_sigmoid_labels = output_sigmoid_labels
        self.output_labels = output_labels
        
        self.output_size = output_sigmoid_classes + (sum(output_classes) if len(output_classes) > 0 else 0)
        
        self.params = []
        self.indexed_params = []
        
        self.create_shared_variables()
        self._create_update_fun()
    
    def create_shared_variables(self):
        """
        Create the theano variables:
        ----------------------------
        
        1. the linear form taking all the words in a window, and
        2. the bilinear form tensor
        3. adding a bias vector, to get an output class prediction of 'output_size'
           size.
           
        """
        
        # in the typical case we consider a linear projection of the concatenation
        # or addition of inputs:
        if self.concatenate:
            projection_size = self.size * self.window + self.object_size
        else:
            assert(self.size == self.object_size), "In additive (not concatenated) projection, "
            "the hidden objects must have the same dimensionality as the language model vectors."
            projection_size = self.size
            
        
        # create linear form matrix:
        proj_matrix = ((1.0 / self.size) * \
            np.random.randn(
                self.output_size,
                projection_size).astype(REAL))
        self.projection_matrix = theano.shared(proj_matrix, name='projection_matrix')    
        self.params.append(self.projection_matrix)
        
        # in the case we want to use 2nd order behavior for modeling we add a tensor:
        if self.bilinear_form:
            # create bilinear form matrix:
            random_comp_matrix = ((1.0 / (self.size)) * np.random.randn(
                self.output_size,
                projection_size,
                projection_size)).astype(REAL)
            self.composition_matrix = theano.shared(random_comp_matrix, name='composition_matrix')
            
            self.params.append(self.composition_matrix)
        
        
        
        # create bias vector:
        bias_vector      = np.zeros(self.output_size, dtype=REAL)
        self.bias_vector = theano.shared(bias_vector, name='bias_vector')
        self.params.append(self.bias_vector)
        
        model_matrix = ((1.0 / self.size) * \
                        np.random.randn(self.vocabulary_size, self.size)).astype(REAL)
        self.model_matrix = theano.shared(model_matrix, name='model_matrix')
        self.params.append(self.model_matrix)
        
        self.indexed_params.append(self.model_matrix)
        
        object_matrix = ((1.0 / self.object_size) * \
                        np.random.randn(self.object_vocabulary_size, self.object_size)).astype(REAL)
        self.object_matrix = theano.shared(object_matrix, name='object_matrix')
        self.params.append(self.object_matrix)
        
        self.indexed_params.append(self.object_matrix)

    def create_normalized_matrices(self):
        self.norm_model_matrix = (self.model_matrix.get_value(borrow=True) / np.sqrt((self.model_matrix.get_value(borrow=True) ** 2).sum(-1))[..., np.newaxis]).astype(REAL)
        self.norm_object_matrix = (self.object_matrix.get_value(borrow=True) / np.sqrt((self.object_matrix.get_value(borrow=True) ** 2).sum(-1))[..., np.newaxis]).astype(REAL)

    def most_similar_word(self, word, topn = 10):
        index = self.vocab.get_index(word)
        word = self.norm_model_matrix[index]
        dists = np.dot(self.norm_model_matrix, word).astype(REAL)
        best = np.argsort(dists)[::-1][:topn + 1]
        result = [(self.vocab.index2word[sim], float(dists[sim]), sim) for sim in best if sim != index]
        return result[:topn]

    def most_similar_object(self, object_index, topn = 20):
        if object_index is np.ndarray or list:
            object = object_index
        else:
            object = self.norm_object_matrix[object_index]
            topn = topn + 1
        dists = np.dot(self.norm_object_matrix, object).astype(REAL)
        best = np.argsort(dists)[::-1][:topn]
        if object_index is np.ndarray or list:
            result = [(sim, float(dists[sim])) for sim in best]
        else:
            result = [(sim, float(dists[sim])) for sim in best if sim != object_index]
        return result[:topn]


    def projection_fun(self, observation, labels, sigmoid_labels):
        if self.bilinear_form:
            log_unnormalized_pred = bilinear_form(self.composition_matrix, observation) + T.dot(self.projection_matrix, observation) + self.bias_vector
        else:
            log_unnormalized_pred = T.dot(self.projection_matrix, observation)  + self.bias_vector
        
        prev_position = 0
        preds = []
        
        error = 0.0
        for i, class_size in enumerate(self.output_classes):
            class_prediction = T.nnet.softmax(log_unnormalized_pred[prev_position:prev_position + class_size])
            preds.append(class_prediction[0])
            prev_position += class_size
            error += T.nnet.categorical_crossentropy(class_prediction, labels[i:i+1]).sum()
        
        prediction = T.nnet.sigmoid(log_unnormalized_pred[prev_position:])
        error += T.nnet.binary_crossentropy(prediction, sigmoid_labels).sum()

        return (preds, prediction, error)
    
    def _create_update_fun(self):
        """
            Given examples of the form:
                
                (
                    [1100, 1200, 12],
                    [1, 2, 0, 0, 1, 0]
                )
                
            Corresponding to a sequence of words 1100, 1200 and an object 12,
            with labels for class 1, 1, class 2, 2, and for the sigmoid classes
            the third class active, we can do regression.
            
        """
        
        input = T.ivector('input')
        input_object = T.iscalar('input_object_index')
        labels = T.ivector('labels')
        sigmoid_labels = T.ivector('sigmoid_labels')
        
        embeddings = self.model_matrix[input]
        object_embedding = self.object_matrix[input_object]
        
        if self.concatenate:
            # or we concatenate all the words and add the object to it
            merged_embeddings = T.concatenate([embeddings.ravel(), object_embedding])
        else:
            # or we sum all the words and add the object to it:
            merged_embeddings = embeddings.sum(axis=1) + object_embedding

        preds, prediction, error = self.projection_fun(merged_embeddings, labels, sigmoid_labels)

        updates = OrderedDict()
        
        gparams = T.grad(error, self.params)
        
        for gparam, param in zip(gparams, self.params):
            if param == self.model_matrix:
                updates[param] = T.inc_subtensor(param[input], - self.alpha * gparam[input])
            elif param == self.object_matrix:
                updates[param] = T.inc_subtensor(param[input_object], - self.alpha * gparam[input_object])
            else:
                updates[param] = param - self.alpha * gparam
            
        self.predict_proba = theano.function([input, input_object], preds + [prediction], mode = self.theano_mode)
        self.predict = theano.function([input, input_object], [pred.argmax() for pred in preds] + [prediction.round()], mode = self.theano_mode)

        input_vector = T.vector()
        alt_preds, alt_prediction, alt_error = self.projection_fun(input_vector, labels, sigmoid_labels)
        self.predict_vector = theano.function([input_vector], [pred.argmax() for pred in alt_preds] + [alt_prediction.round()], mode = self.theano_mode)

        self.predict_vector_proba = theano.function([input_vector], alt_preds + [alt_prediction], mode = self.theano_mode)
        
        training_inputs = []
        if len(self.output_classes) > 0:
            training_inputs.append(labels)
        if self.output_sigmoid_classes > 0:
            training_inputs.append(sigmoid_labels)
        
        self.gradient_fun  = theano.function([input, input_object] + training_inputs, gparams, mode = self.theano_mode)
        self.update_fun    = theano.function([input, input_object] + training_inputs, error, updates = updates, mode = self.theano_mode)

    def predict(self, indices, object_index):
        return predict_sentence_window(self, indices, object_index)

    def predict_proba(self, indices, object_index):
        return predict_distribution_sentence_window(self, indices, object_index)

    def train(self, texts, chunksize=100, workers = 2):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of utf8 strings.

        """
        logger.info("training model with %i workers" % (workers))

        start, next_report = time.time(), [1.0]
        jobs = Queue(maxsize=2 * workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

        total_error = [0.0]
        objects_done = [0]

        def worker_train():
            """Train the model, lifting lists of sentences from the jobs queue."""
            observation_work = np.zeros(self.window * self.size + self.object_size, dtype = REAL)
            prediction_work = np.zeros(self.output_size, dtype = REAL)

            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                # how many words did we train on? out-of-vocabulary (unknown) words do not count
                error = sum(train_sentence_concatenation(self, sentence, object_index, softmax_target, sigmoid_target, self._alpha, prediction_work, observation_work) for sentence, object_index, softmax_target, sigmoid_target in job)
                with lock:
                    total_error[0] += error
                    objects_done[0] += len(job)
                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        logger.info("PROGRESS: %s objects, %.0f objects/s" % (objects_done[0], float(objects_done[0]) / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

        dynos = [threading.Thread(target=worker_train) for _ in range(0,workers)]
        for thread in dynos:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        # convert input strings to Vocab objects (or None for OOV words), and start filling the jobs queue
        no_oov = ((np.array([self.vocab.get_index(word) for word in sentence], dtype = INT), object_index, softmax_target, sigmoid_target) for sentence, object_index, softmax_target, sigmoid_target in texts)
        for job_no, job in enumerate(gensim_utils.grouper(no_oov, chunksize)):
            logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
            jobs.put(job)
        logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())

        for _ in range(0,workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in dynos:
            thread.join()

        elapsed = time.time() - start
        logger.info("training on %i objects took %.1fs, %.0f words/s" %
            (objects_done[0], elapsed, objects_done[0] / elapsed if elapsed else 0.0))

        return (objects_done[0], total_error[0])

class CategoriesConverter(object):
    """
    Sorts and converts named lists into
    binary vectors.
    
    """
    def __init__(self, categories):
        self.num_categories = len(categories)
        self.index2category = list(categories)
        
        vocab = {}
        for i, cat in enumerate(self.index2category):
            vocab[cat] = i
        self.category2index = vocab
        
    def convert_categories_to_label(self, categories):
        label = np.zeros(self.num_categories, dtype=INT)
        
        for cat in categories:
            label[self.category2index[cat]] = 1
            
        return label        
    
    def save(self, path):
        with gzip.open(path, 'wb') as file:
            pickle.dump(self, file, 1)

    def save_to_java(self, path):
        with gzip.open(path, 'wt') as file:
            for category in self.index2category:
                file.write((category + "\n"))
            
    @staticmethod
    def load(path):
        file = gzip.open(path, 'r')
        self = pickle.load(file)
        file.close()
        return self

class DatasetGenerator():
    def __init__(self, texts, texts_data, category_converter):
        self.texts = texts
        self.texts_data = texts_data
        self.category_converter = category_converter

    def save_ids(self, path):
        with gzip.open(path, "wt") as f:
            for datum in self.texts_data:
                f.write( (datum["_id"] + "\n") )

    def save(self, path):
        with gzip.open(path, "wt") as f:
            json.dump(self.texts_data, f)

        
    def convert_datum_to_classes(self, datum):
        return np.array([len(datum["price"]), max(0, int(datum["rating"])-1)], dtype=INT)
    
    def __iter__(self):
        for i, text in enumerate(self.texts):
            yield((text,
                   i,
                   self.convert_datum_to_classes(self.texts_data[i]),
                   self.category_converter.convert_categories_to_label(self.texts_data[i]["categories"])))