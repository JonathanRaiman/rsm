from objectlm import ObjectLM

# should modify this to do some auto-encoding / self regression.

from word2vec_extended import Word2VecExtended

import sys, numpy as np
# to be able to pickle load the model below:
sys.path.append("/Users/jonathanraiman/Desktop/Coding/language_modeling/")

lmsenti = Word2VecExtended.load("/Users/jonathanraiman/Desktop/Coding/language_modeling/saves/kid_model_30_oov_senti")

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from .train_model import get_proj_matrix_grad

model = ObjectLM(
    vocabulary = lmsenti,
    object_vocabulary_size = 200,
    window = 10,
    bilinear_form = False,
    size = 20,
    object_size = 20,
    output_sigmoid_classes = 20,
    output_classes=[5, 5] # "", "$", "$$",...,"$$$$", 5 price classes, and 5 rating classes
)


window      = model.window
size        = model.size
object_size = model.size
object_vocab_size = model.object_vocabulary_size
vocab_size        = model.vocabulary_size

INT_t = np.int32

# create target distributions:
num_sigmoid_classes = model.output_sigmoid_classes
distribution_sizes = model.output_classes
num_softmax_classes = len(distribution_sizes)

# target size is found by getting number of sigmoid units + number of softmax outputs needed
softmax_target_size = sum(distribution_sizes)
target_size = softmax_target_size + num_sigmoid_classes

indices = np.random.randint(0, vocab_size, window).astype(np.int32)
object_index = np.random.randint(0, object_vocab_size)
sigmoid = lambda x: (1. / (1. + np.exp(-x))).astype(np.float32)
softmax = lambda x: np.exp(x) / np.exp(x).sum()
softmax_error = lambda target, distribution: -np.log(distribution[target])
def binary_crossentropy_error(target, prob):
    if target == 0:
        return - np.log(1. - prob)
    else:
        return - np.log(prob)
    
def binary_crossentropy_error_vector(targets, probs):
    return sum(binary_crossentropy_error(target, probs[i]) for i, target in enumerate(targets))

softmax_target = np.array([np.random.randint(0, class_size) for class_size in distribution_sizes], dtype=INT_t)
sigmoid_target = np.random.randint(0, 2, num_sigmoid_classes).astype(INT_t)

proj_mat      = model.projection_matrix.get_value(borrow=True)
model_matrix  = model.model_matrix.get_value(borrow=True)
object_matrix = model.object_matrix.get_value(borrow=True)

distribution_work = np.zeros(target_size, dtype=np.float32)
observation_work = np.zeros(window * size + object_size, dtype=np.float32)

def using_cython():
    return get_proj_matrix_grad(
            indices,
            object_index,
            softmax_target,
            sigmoid_target,
            proj_mat,
            model_matrix,
            object_matrix,
            distribution_sizes,
            window,
            softmax_target_size,
            num_sigmoid_classes)
    
def using_numpy():
    # for testing:
    obs_2_input = np.zeros(window * size + object_size, dtype=np.float32)
    for k, i in enumerate(indices):
        obs_2_input[k * size: (k + 1) * size] = model_matrix[i]
    obs_2_input[size * window: size * window + object_size] = object_matrix[object_index]
    
    # multiply with matrix to get projection:
    obs_2 = np.dot(proj_mat, obs_2_input)
    
    # use softmax to get exponential normalization:
    class_offset = 0
    total_error = np.float32(0.0)
    for i in range(num_softmax_classes):
        obs_2[class_offset:class_offset + distribution_sizes[i]] = softmax(obs_2[class_offset:class_offset + distribution_sizes[i]])
        total_error += softmax_error(softmax_target[i], obs_2[class_offset:class_offset + distribution_sizes[i]])
        class_offset += distribution_sizes[i]
        
    obs_2[class_offset :] = sigmoid(obs_2[class_offset :])
    
    for i in range(0, num_sigmoid_classes):
        total_error += binary_crossentropy_error(sigmoid_target[i], obs_2[class_offset + i])
    
    delta_class = obs_2.copy()
    delta_class[class_offset :] -= sigmoid_target
    
    class_offset = 0
    for i in range(num_softmax_classes):
        delta_class[class_offset + softmax_target[i]] -= 1
        class_offset += distribution_sizes[i]
        
    proj_mat_grad = np.outer(delta_class, obs_2_input)
    bias_vec_grad = delta_class
    
    lm_grad = np.dot(proj_mat.T, delta_class)
    
    lm_grad_holder = np.zeros_like(model_matrix)
    lm_grad_holder[indices] += lm_grad[0:window * size].reshape([window, size])
    
    object_lm_grad_holder = np.zeros_like(object_matrix)
    object_lm_grad_holder[object_index] += lm_grad[window * size:]
    
    return (proj_mat_grad, bias_vec_grad, lm_grad_holder, object_lm_grad_holder)

def using_theano():
    return model.gradient_fun(indices, object_index, softmax_target, sigmoid_target)

print(all(np.allclose(a,b) for a, b in zip(using_cython(), using_numpy())) and all(np.allclose(a,b) for a, b in zip(using_cython(), using_theano())))
