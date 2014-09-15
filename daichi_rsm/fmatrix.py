#!/usr/local/bin/python
"""
fmatrix.py: loading sparse feature matrix.
$Id: fmatrix.py,v 1.1 2013/06/28 08:06:00 daichi Exp $

"""

import sys
import numpy as np
from theano import config as theano_config

def dstat(file):
    """
    returns documents and words in 'id:cnt' data.
    input: path to training file,
    output: (d, v)
        d the number of documents, and
        v the size of the vocabulary.
    """
    d,v = 0,0
    with open (file, 'r') as fh:
        for line in fh:
            tokens = line.split()
            if len(tokens) > 0:
                d = d + 1
                for token in tokens:
                    [id,cnt] = token.split(':')
                    if int(id) > v:
                        v = int(id)
    return (d,v)

def parse (file):
    """
    build a numpy full matrix from sparse 'id:cnt' data.
    """
    [docs,words] = dstat(file)
    d = 0
    matrix = np.zeros((docs, words), dtype=theano_config.floatX)
    with open (file, 'r') as fh:
        for line in fh:
            tokens = line.split()
            if len(tokens) > 0:
                for token in tokens:
                    [id,cnt] = token.split(':')
                    v = int(id) - 1
                    c = float(cnt)
                    matrix[d,v] = c
                d = d + 1
                
    return matrix
