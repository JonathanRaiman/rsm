# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp, log
from libc.string cimport memset
from libc.float cimport FLT_MAX

from cpython cimport PyCapsule_GetPointer # PyCObject_AsVoidPtr
from scipy.linalg.blas import fblas

REAL = np.float32
ctypedef np.float32_t REAL_t
ctypedef np.int32_t  INT_t
ctypedef np.int32_t  LABEL_INT

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0
cdef REAL_t ZEROF = <REAL_t>0.0
cdef REAL_t SMALL_NUM = <REAL_t>1e-6

ctypedef void (*sger_ptr) (const int *M, const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY, float *A, const int * LDA) nogil
ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil
ctypedef void (*sgemv_ptr) (char *trans, int *m, int *n,\
                     float *alpha, float *a, int *lda, float *x,\
                     int *incx,\
                     float *beta,  float *y, int *incy) nogil

cdef sgemv_ptr sgemv=<sgemv_ptr>PyCapsule_GetPointer(fblas.cblas.sgemv._cpointer, NULL) # y := A*x + beta * y
cdef sger_ptr sger=<sger_ptr>PyCapsule_GetPointer(fblas.sger._cpointer , NULL)  # A := alpha*x*y.T + A
cdef scopy_ptr scopy=<scopy_ptr>PyCapsule_GetPointer(fblas.scopy._cpointer , NULL)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCapsule_GetPointer(fblas.saxpy._cpointer , NULL)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCapsule_GetPointer(fblas.sdot._cpointer     , NULL)      # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCapsule_GetPointer(fblas.sdot._cpointer  , NULL)   # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCapsule_GetPointer(fblas.snrm2._cpointer, NULL) # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCapsule_GetPointer(fblas.sscal._cpointer, NULL) # x = alpha * x

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6
cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef char trans  = 'T'
cdef char transN = 'N'

def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.

    """
    cdef int i
    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
        
    return 0

def outer_prod(_x, _y, _output):
    cdef REAL_t *x = <REAL_t *>(np.PyArray_DATA(_x))
    cdef int M = _y.shape[0]
    cdef int N = _x.shape[0]
    cdef REAL_t *y = <REAL_t *>(np.PyArray_DATA(_y))
    cdef REAL_t *output = <REAL_t *>(np.PyArray_DATA(_output))
    sger(&M, &N, &ONEF, y, &ONE, x, &ONE, output, &M)

cdef void outer_product_alpha(REAL_t * alpha, int *asize, REAL_t *a, int *bsize, REAL_t *b, REAL_t *output) nogil:
    sger(bsize, asize, alpha, b, &ONE, a, &ONE, output, bsize)

cdef void outer_product(int *asize, REAL_t *a, int *bsize, REAL_t *b, REAL_t *output) nogil:
    sger(bsize, asize, &ONEF, b, &ONE, a, &ONE, output, bsize)
    
cdef void copy_vector(REAL_t *source, int *copy_size, REAL_t *target) nogil:
    scopy(copy_size, source, &ONE, target, &ONE)

cdef void add_vector(REAL_t *source, int *copy_size, REAL_t *target) nogil:
    saxpy(copy_size, &ONEF, source, &ONE, target, &ONE)

cdef void add_vector_alpha(REAL_t * alpha, REAL_t *source, int *copy_size, REAL_t *target) nogil:
    saxpy(copy_size, alpha, source, &ONE, target, &ONE)
    
cdef void matrix_vector_product(char * tranposed, int *M, int *N, REAL_t *matrix, REAL_t *vector, REAL_t* destination) nogil:
    sgemv(tranposed, N, M, &ONEF, matrix, N, vector, &ONE, &ZEROF, destination, &ONE)

def matrix_product(A, B):
    cdef int M = A.shape[0]
    cdef int N = A.shape[1]
    
    if B.shape[0] != N:
        raise ValueError("matrices are not aligned")
    
    cdef np.ndarray C = np.zeros(M, dtype=REAL)
    
    matrix_vector_product(&trans, &M, &N, <REAL_t *>(np.PyArray_DATA(A)), <REAL_t *>(np.PyArray_DATA(B)), <REAL_t *>(np.PyArray_DATA(C)))
    return C

cdef void softmax(float *observation, LABEL_INT * observation_size, float * softmax_destination) nogil:
    cdef Py_ssize_t i
    cdef float exp_sum = 0.0
    for i in range(0, observation_size[0]):
        softmax_destination[i] = exp(observation[i])
        exp_sum += softmax_destination[i]
    for i in range(0, observation_size[0]):
        softmax_destination[i] /= exp_sum

cdef int argmax(int * distribution_size, REAL_t * distribution) nogil:
    cdef Py_ssize_t i
    cdef float largest_prob = -FLT_MAX
    cdef int index = 0
    for i in range(0, distribution_size[0]):
        if distribution[i] > largest_prob:
            largest_prob = distribution[i]
            index = i
    return index

cdef float sigmoid(float x) nogil:
    return 1./ (1. + exp(-x))
    
    # for faster processing:
    #if x <= -MAX_EXP:
    #    return ZEROF
    #if x >= MAX_EXP:
    #    return ONEF
    #return EXP_TABLE[<int>((x + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

cdef float binary_crossentropy_error(INT_t target, float prob) nogil:
    cdef INT_t zero_target = 0
    if target == zero_target:
        return -log(1. - prob)
    else:
        return -log(prob)
        
cdef float softmax_error(INT_t * target, float * observations) nogil:
    return -log(observations[target[0]])

cdef float binary_crossentropy_error_vector(np.uint32_t * target, int * sigmoid_target_size, float * probs) nogil:
    cdef Py_ssize_t i
    cdef float total_error = 0.
    for i in range(0, sigmoid_target_size[0]):
        total_error += binary_crossentropy_error(target[i], probs[i])
    return total_error
    
cdef void a_plus_alpha_b_equals_c(int * size, float *a, float * alpha, float *b, float *c) nogil:
    # copy a to c
    scopy(size, a, &ONE, c, &ONE)
    # inplace add alpha * b to c
    saxpy(size, alpha, b, &ONE, c, &ONE)

def get_binary_cross_error(_targets, _distribution):
    cdef REAL_t * distribution = <REAL_t *>(np.PyArray_DATA(_distribution))
    cdef np.uint32_t * targets = <np.uint32_t *>(np.PyArray_DATA(_targets))
    cdef int num_sigmoid_classes = len(_targets)
    return binary_crossentropy_error_vector(targets, &num_sigmoid_classes, distribution)

# core training system:
def train_sentence_concatenation(model,
    _indices,
    int object_index,
    _softmax_target,
    _sigmoid_target,
    REAL_t alpha,
    _distribution_work,
    _observation_work):
    """
    Train the model using a block of text represented
    by indices to make predictions over multimodal and
    unimodal (sigmoid) classes.

    Inputs
    ------

    <np.array int32>             _indices : the words' indices
    <int>                    object_index : the object's (full text's) index
    <np.array int32>      _softmax_target : softmax labels (multimodal)
    <np.array int32>      _sigmoid_target : sigmoid labels (unimodal / binary)
    <float>                         alpha : the learning rate
    <np.array float32> _distribution_work : empty array to make calculations
                                            on prediction space size
    <np.array float32>  _observation_work : empty array to make calculations
                                            on observation space size

    Outputs
    -------

    <double> error : total error incured during this training sequence.

    """
    cdef Py_ssize_t i
    # model parameters:
    cdef int size                  = model.size
    cdef int object_size           = model.object_size
    cdef int window                = model.window

    # theano variables (with underlying numpy matrices)
    cdef REAL_t *projection_matrix = <REAL_t *>(np.PyArray_DATA(model.projection_matrix.get_value(borrow=True)))
    cdef REAL_t *bias_vector       = <REAL_t *>(np.PyArray_DATA(model.bias_vector.get_value(borrow=True)))
    cdef REAL_t *model_matrix      = <REAL_t *>(np.PyArray_DATA(model.model_matrix.get_value(borrow=True)))
    cdef REAL_t *object_matrix     = <REAL_t *>(np.PyArray_DATA(model.object_matrix.get_value(borrow=True)))

    cdef REAL_t *distribution_work = <REAL_t *>(np.PyArray_DATA(_distribution_work))
    cdef REAL_t *observation_work  = <REAL_t *>(np.PyArray_DATA(_observation_work))

    # observation parameters:
    cdef int observation_size    = window * size + object_size

    # regression parameters:
    cdef int num_softmax_classes = len(model.output_classes)
    cdef int num_sigmoid_classes = model.output_sigmoid_classes
    cdef LABEL_INT *distribution_sizes = <LABEL_INT *>(np.PyArray_DATA(model.output_classes))
    
    cdef int softmax_target_size = 0
    for i in range(0, num_softmax_classes):
        softmax_target_size += distribution_sizes[i]

    # regression inputs:
    cdef INT_t * indices        = <INT_t *>(np.PyArray_DATA(_indices))
    cdef INT_t * softmax_target = <INT_t *>(np.PyArray_DATA(_softmax_target))
    cdef INT_t * sigmoid_target = <INT_t *>(np.PyArray_DATA(_sigmoid_target))
    cdef int     text_len       = np.PyArray_DIM(_indices, 0)
    cdef double error = 0.0
    
    with nogil:
        for i in range(text_len+1-window):
            error += train_sentence_concatenation_window(
                &alpha,
                projection_matrix,
                bias_vector,
                model_matrix,
                object_matrix,
                observation_work,
                distribution_work,
                distribution_sizes,
                softmax_target,
                sigmoid_target,
                &softmax_target_size,
                &num_softmax_classes,
                &num_sigmoid_classes,
                &observation_size,
                &size,
                &object_size,
                &window,
                &indices[i],
                &object_index)
    return error

cdef void create_observation_vector(
    INT_t * indices,
    int * object_index,
    int * window,
    int * size,
    int * object_size,
    REAL_t * model_matrix,
    REAL_t * object_matrix,
    REAL_t * work) nogil:

    cdef Py_ssize_t i
    # for each element in the window we copy a word:
    for i in range(0, window[0]):
        copy_vector(&model_matrix[indices[i] * size[0]], size, &work[i * size[0]])
       
    # we also copy the object vector:
    copy_vector(&object_matrix[object_index[0] * object_size[0]], object_size, &work[window[0] * size[0]])

cdef void make_prediction(
    int * observation_size,
    int * target_size,
    LABEL_INT *distribution_sizes,
    INT_t * softmax_target,
    INT_t * sigmoid_target,
    int * num_softmax_classes,
    int * num_sigmoid_classes,
    REAL_t *projection_matrix,
    REAL_t *bias_vector,
    REAL_t *observation,
    REAL_t *output,
    REAL_t *error) nogil:

    # multiply with vectors to get projection:
    matrix_vector_product(&trans, target_size, observation_size, projection_matrix, observation, output)

    # add the bias vector
    add_vector(bias_vector, target_size, output)

    # softmax and sigmoid output:
    cdef Py_ssize_t i
    cdef int class_offset = 0
    
    # for all softmax classes:
    for i in range(0, num_softmax_classes[0]):
        softmax(&output[class_offset], &distribution_sizes[i], &output[class_offset])
        if output[class_offset + softmax_target[i]] > SMALL_NUM:
            error[0] += softmax_error(&softmax_target[i], &output[class_offset])
        class_offset += distribution_sizes[i]

    # for all sigmoid classes:
    for i in range(0, num_sigmoid_classes[0]):
        if output[class_offset + i] > -MAX_EXP and output[class_offset + i] < MAX_EXP:
            output[class_offset + i] = sigmoid(output[class_offset + i])
            error[0] += binary_crossentropy_error(sigmoid_target[i], output[class_offset + i])
        else:
            # error will cause craziness to ensue.
            output[class_offset + i] = sigmoid(output[class_offset + i])

cdef void make_prediction_no_error(
    int * observation_size,
    int * target_size,
    LABEL_INT *distribution_sizes,
    int * num_softmax_classes,
    int * num_sigmoid_classes,
    REAL_t *projection_matrix,
    REAL_t *bias_vector,
    REAL_t *observation,
    REAL_t *output) nogil:

    # multiply with vectors to get projection:
    matrix_vector_product(&trans, target_size, observation_size, projection_matrix, observation, output)

    # add the bias vector
    add_vector(bias_vector, target_size, output)

    # softmax and sigmoid output:
    cdef Py_ssize_t i
    cdef int  class_offset = 0
    
    # for all softmax classes:
    for i in range(0, num_softmax_classes[0]):
        softmax(&output[class_offset], &distribution_sizes[i], &output[class_offset])
        class_offset += distribution_sizes[i]

    # for all sigmoid classes:
    for i in range(0, num_sigmoid_classes[0]):
        output[class_offset + i] = sigmoid(output[class_offset + i])

cdef void obtain_delta_class(
    INT_t * softmax_target,
    INT_t * sigmoid_target,
    int * num_softmax_classes,
    int * num_sigmoid_classes,
    LABEL_INT * distribution_sizes,
    REAL_t * distribution
    ) nogil:
    """
    Create the delta_class in-place, e.g. the
    vector of errors in each dimension for each
    activation, with t the labels, and y the
    prediction:

        delta_class = y - t

    """
    
    cdef Py_ssize_t i
    cdef int class_offset = 0

    for i in range(0, num_softmax_classes[0]):
        distribution[class_offset + softmax_target[i]] = distribution[class_offset + softmax_target[i]] - ONEF
        class_offset += distribution_sizes[i]
        
    for i in range(0, num_sigmoid_classes[0]):
        distribution[class_offset + i] = distribution[class_offset + i] - sigmoid_target[i]

cdef void update_bias_vector(
    int * target_size,
    REAL_t *bias_vector,
    REAL_t *alpha,
    REAL_t *delta_class
    ) nogil:
    """
    The bias vector's update rule is
    to subtract the delta_class * alpha
    from its current value.

    """

    cdef REAL_t neg_alpha = -alpha[0]

    saxpy(target_size, &neg_alpha, delta_class, &ONE, bias_vector, &ONE)

cdef void update_projection_matrix(
    int * observation_size,
    int * target_size,
    REAL_t * projection_matrix,
    REAL_t * alpha,
    REAL_t * observation,
    REAL_t * delta_class
    ) nogil:
    """
    The update rule for the projection matrix
    is to take the outer product of the observation
    and the delta_class, multiplied by alpha, and
    substract this matrix from the current
    projection matrix.

    """

    cdef float neg_alpha = -alpha[0]
    outer_product_alpha(&neg_alpha, target_size, delta_class, observation_size, observation, projection_matrix)

cdef void update_language_model(
    int * observation_size,
    int * target_size,
    int * size,
    int * object_size,
    int * window,
    INT_t * indices,
    int * object_index,
    REAL_t * projection_matrix,
    REAL_t * model_matrix,
    REAL_t * object_matrix,
    REAL_t * alpha,
    REAL_t * delta_class,
    REAL_t * work) nogil:

    cdef REAL_t neg_alpha = -alpha[0]
    cdef Py_ssize_t i

    # obtain the projected version of the error in the observation space
    matrix_vector_product(&transN, target_size, observation_size, projection_matrix, delta_class, work)
    
    # obtain the word vector updates:
    for i in range(0, window[0]):
        add_vector_alpha(&neg_alpha, &work[i * size[0]], size, &model_matrix[indices[i] * size[0]])
    
    # copy the object vector over
    add_vector_alpha(&neg_alpha, &work[window[0] * size[0]], object_size, &object_matrix[object_index[0] * object_size[0]])

def predict_sentence_window(model,
    _indices,
    int object_index):
    cdef Py_ssize_t i
    # model parameters:
    cdef int size                  = model.size
    cdef int object_size           = model.object_size
    cdef int window                = model.window

    # theano variables (with underlying numpy matrices)
    cdef REAL_t *projection_matrix = <REAL_t *>(np.PyArray_DATA(model.projection_matrix.get_value(borrow=True)))
    cdef REAL_t *bias_vector       = <REAL_t *>(np.PyArray_DATA(model.bias_vector.get_value(borrow=True)))
    cdef REAL_t *model_matrix      = <REAL_t *>(np.PyArray_DATA(model.model_matrix.get_value(borrow=True)))
    cdef REAL_t *object_matrix     = <REAL_t *>(np.PyArray_DATA(model.object_matrix.get_value(borrow=True)))

    # observation parameters:
    cdef int observation_size    = window * size + object_size

    # prediction parameters:
    cdef int num_softmax_classes = len(model.output_classes)
    cdef int num_sigmoid_classes = model.output_sigmoid_classes
    cdef LABEL_INT *distribution_sizes = <LABEL_INT *>(np.PyArray_DATA(model.output_classes))

    cdef int target_size = model.output_size

    cdef np.ndarray _distribution_work = np.zeros(target_size, dtype = REAL)
    cdef REAL_t *distribution_work = <REAL_t *>(np.PyArray_DATA(_distribution_work))
    cdef np.ndarray _observation_work = np.zeros(observation_size, dtype = REAL)
    cdef REAL_t *observation_work  = <REAL_t *>(np.PyArray_DATA(_observation_work))

    # prediction inputs:
    cdef INT_t * indices        = <INT_t *>(np.PyArray_DATA(_indices))

    create_observation_vector(indices, &object_index, &window, &size, &object_size, model_matrix, object_matrix, observation_work)

    make_prediction_no_error(&observation_size, &target_size, distribution_sizes, &num_softmax_classes, &num_sigmoid_classes, projection_matrix, bias_vector, observation_work, distribution_work)

    cdef np.ndarray _softmax_predictions = np.zeros(num_softmax_classes, dtype = np.int32)
    cdef INT_t * softmax_predictions = <INT_t *>np.PyArray_DATA(_softmax_predictions)
    cdef np.ndarray _sigmoid_predictions = np.zeros(num_sigmoid_classes, dtype = np.int32)
    cdef INT_t * sigmoid_predictions = <INT_t *>np.PyArray_DATA(_sigmoid_predictions)

    cdef int class_offset = 0
    cdef int distribution_size = 0
    for i in range(0, num_softmax_classes):
        distribution_size = distribution_sizes[i]
        softmax_predictions[i] = argmax(&distribution_size, &distribution_work[class_offset])
        class_offset += distribution_sizes[i]

    for i in range(0, num_sigmoid_classes):
        sigmoid_predictions[i] = round(distribution_work[class_offset+i])

    return (_softmax_predictions, _sigmoid_predictions)

def predict_distribution_sentence_window(model,
    _indices,
    int object_index):
    cdef Py_ssize_t i
    # model parameters:
    cdef int size                  = model.size
    cdef int object_size           = model.object_size
    cdef int window                = model.window

    # theano variables (with underlying numpy matrices)
    cdef REAL_t *projection_matrix = <REAL_t *>(np.PyArray_DATA(model.projection_matrix.get_value(borrow=True)))
    cdef REAL_t *bias_vector       = <REAL_t *>(np.PyArray_DATA(model.bias_vector.get_value(borrow=True)))
    cdef REAL_t *model_matrix      = <REAL_t *>(np.PyArray_DATA(model.model_matrix.get_value(borrow=True)))
    cdef REAL_t *object_matrix     = <REAL_t *>(np.PyArray_DATA(model.object_matrix.get_value(borrow=True)))

    # observation parameters:
    cdef int observation_size    = window * size + object_size

    # prediction parameters:
    cdef int num_softmax_classes = len(model.output_classes)
    cdef int num_sigmoid_classes = model.output_sigmoid_classes
    cdef LABEL_INT *distribution_sizes = <LABEL_INT *>(np.PyArray_DATA(model.output_classes))

    cdef int target_size = model.output_size

    cdef np.ndarray _distribution_work = np.zeros(target_size, dtype = REAL)
    cdef REAL_t *distribution_work = <REAL_t *>(np.PyArray_DATA(_distribution_work))
    cdef np.ndarray _observation_work = np.zeros(observation_size, dtype = REAL)
    cdef REAL_t *observation_work  = <REAL_t *>(np.PyArray_DATA(_observation_work))

    # prediction inputs:
    cdef INT_t * indices        = <INT_t *>(np.PyArray_DATA(_indices))

    create_observation_vector(indices, &object_index, &window, &size, &object_size, model_matrix, object_matrix, observation_work)

    make_prediction_no_error(&observation_size, &target_size, distribution_sizes, &num_softmax_classes, &num_sigmoid_classes, projection_matrix, bias_vector, observation_work, distribution_work)

    return _distribution_work

cdef REAL_t train_sentence_concatenation_window(
    const REAL_t * alpha,
    REAL_t *projection_matrix,
    REAL_t *bias_vector,
    REAL_t *model_matrix,
    REAL_t *object_matrix,
    REAL_t *observation_work,
    REAL_t *distribution_work,
    LABEL_INT *distribution_sizes,
    INT_t * softmax_target,
    INT_t * sigmoid_target,
    const int * softmax_target_size,
    const int * num_softmax_classes,
    const int * num_sigmoid_classes,
    const int * observation_size,
    const int * size,
    const int * object_size,
    const int * window,
    INT_t * indices,
    int * object_index) nogil:

    cdef int target_size = softmax_target_size[0] + num_sigmoid_classes[0]
    cdef REAL_t total_error = 0.
    memset(distribution_work, 0, target_size * cython.sizeof(REAL_t))

    create_observation_vector(indices, object_index, window, size, object_size, model_matrix, object_matrix, observation_work)

    make_prediction(observation_size, &target_size, distribution_sizes, softmax_target, sigmoid_target, num_softmax_classes, num_sigmoid_classes, projection_matrix, bias_vector, observation_work, distribution_work, &total_error)

    obtain_delta_class(softmax_target, sigmoid_target, num_softmax_classes, num_sigmoid_classes, distribution_sizes, distribution_work)

    update_bias_vector(&target_size, bias_vector, alpha, distribution_work)
    update_projection_matrix(observation_size, &target_size, projection_matrix, alpha, observation_work, distribution_work)

    memset(observation_work, 0, observation_size[0] * cython.sizeof(REAL_t))
    update_language_model(observation_size, &target_size, size, object_size, window, indices, object_index, projection_matrix, model_matrix, object_matrix, alpha, distribution_work, observation_work)

    return total_error

def get_proj_matrix_grad(
    _indices,
    int object_index,
    _softmax_target,
    _sigmoid_target,
    _projection_matrix,
    _bias_vector,
    _model_matrix,
    _object_matrix,
    _distribution_sizes,
    int window,
    int softmax_target_size,
    int num_sigmoid_classes):
    
    # => should also convert _word_indices to a np.uint32_t array for speed.
    
    cdef int vocabulary_size = np.PyArray_DIM(_model_matrix, 0)
    cdef int size = np.PyArray_DIM(_model_matrix, 1)
    
    cdef int object_vocabulary_size = np.PyArray_DIM(_object_matrix, 0)
    cdef int object_size = np.PyArray_DIM(_object_matrix, 1)
    
    cdef int observation_size    = window * size + object_size
    cdef int num_softmax_classes = len(_distribution_sizes)
    
    cdef Py_ssize_t i
    cdef np.ndarray observation  = np.zeros( observation_size, dtype=REAL)
    cdef REAL_t *observation_work = <REAL_t *>(np.PyArray_DATA(observation))
    cdef REAL_t *model_matrix    = <REAL_t *>(np.PyArray_DATA(_model_matrix))
    cdef REAL_t *object_matrix   = <REAL_t *>(np.PyArray_DATA(_object_matrix))
    cdef INT_t * softmax_target = <INT_t * >(np.PyArray_DATA(_softmax_target))
    cdef INT_t * sigmoid_target = <INT_t * >(np.PyArray_DATA(_sigmoid_target))
    cdef LABEL_INT *distribution_sizes = <LABEL_INT *>(np.PyArray_DATA(_distribution_sizes))
    
    cdef REAL_t *projection_matrix  = <REAL_t *>(np.PyArray_DATA(_projection_matrix))
    cdef REAL_t *bias_vector        = <REAL_t *>(np.PyArray_DATA(_bias_vector))

    cdef int target_size = softmax_target_size + num_sigmoid_classes
    cdef REAL_t total_error = 0.

    # prediction inputs:
    cdef INT_t * indices        = <INT_t *>(np.PyArray_DATA(_indices))
    
    # for each element in the window we copy a word:

    create_observation_vector(indices, &object_index, &window, &size, &object_size, model_matrix, object_matrix, observation_work)

    cdef np.ndarray destination_distribution = np.zeros(target_size, dtype=REAL)
    cdef REAL_t *distribution_work = <REAL_t *>(np.PyArray_DATA(destination_distribution))

    make_prediction(&observation_size, &target_size, distribution_sizes, softmax_target, sigmoid_target, &num_softmax_classes, &num_sigmoid_classes, projection_matrix, bias_vector, observation_work, distribution_work, &total_error)

    obtain_delta_class(softmax_target, sigmoid_target, &num_softmax_classes, &num_sigmoid_classes, distribution_sizes, distribution_work)
    
    # 2. Outer product with input gets us the gradient for the projection matrix
    
    cdef np.ndarray proj_mat_grad = np.zeros([target_size, observation_size], dtype=REAL)
    cdef REAL_t *proj_mat_grad_ptr = <REAL_t *>(np.PyArray_DATA(proj_mat_grad))
    
    outer_product(&target_size, distribution_work, &observation_size, observation_work, proj_mat_grad_ptr)
    
    # 3. The bias vector is delta class
    
    cdef np.ndarray bias_vec_grad = np.zeros(target_size, dtype=REAL)
    cdef REAL_t *bias_vec_grad_ptr = <REAL_t *>(np.PyArray_DATA(bias_vec_grad))
    scopy(&target_size, distribution_work, &ONE, bias_vec_grad_ptr, &ONE)
    
    # 5. The word vector and object vector gradient are the product of this delta class
    #    by the transpose of the projection matrix.
    
    cdef np.ndarray lm_grad = np.zeros(observation_size, dtype=REAL)
    cdef REAL_t *lm_grad_ptr = <REAL_t *>(np.PyArray_DATA(lm_grad))
    
    matrix_vector_product(&transN, &target_size, &observation_size, projection_matrix, distribution_work, lm_grad_ptr)
    
    cdef np.ndarray lm_grad_holder = np.zeros([vocabulary_size, size], dtype=REAL)
    cdef REAL_t *lm_grad_holder_ptr = <REAL_t *>(np.PyArray_DATA(lm_grad_holder))
    
    for i in range(0, window):
        copy_vector(&lm_grad_ptr[i * size], &size, &lm_grad_holder_ptr[indices[i] * size])
        
    cdef np.ndarray object_lm_grad_holder = np.zeros([object_vocabulary_size, object_size], dtype=REAL)
    cdef REAL_t *object_lm_grad_holder_ptr = <REAL_t *>(np.PyArray_DATA(object_lm_grad_holder))
    
    copy_vector(&lm_grad_ptr[window * size], &object_size, &object_lm_grad_holder_ptr[object_index * object_size])
    
    return (proj_mat_grad, bias_vec_grad, lm_grad_holder, object_lm_grad_holder)

FAST_VERSION = init()