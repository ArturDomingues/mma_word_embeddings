#!/C:\Users\Martin\.conda\envs\martin\python.exe

import numpy as np
from gensim import utils
from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring
import smart_open

def save_word2vec_format_inputs(my_dict):
    '''
    my_dict: a dictionary mapping words to their associated vectors
    For example:
    my_dict = {'white': np.array([0.5,-0.4]), 'black': np.array([0.3,-0.2])
    }
    '''
    
    m = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size =len(list(my_dict.values())[0]))
    m.vocab = my_dict
    m.vectors = np.array(list(my_dict.values()))
    return m.vocab, m.vectors
	
#vocab, vectors = save_word2vec_format_inputs(my_dict) # for extracting the vectors and vocab to feed into my_save_word2vec_format function
	
def my_save_word2vec_format(fname, vocab, vectors, binary=True, total_vec=2):
    
    """Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.

    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vectors : numpy.array
        The vectors to be stored.
    binary : bool, optional
        If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
    total_vec : int, optional
        Explicitly specify total number of vectors
        (in case word vectors are appended with document vectors afterwards).

    """
    if not (vocab or vectors):
        raise RuntimeError("no input")
    if total_vec is None:
        total_vec = len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with smart_open.open(fname, 'wb') as fout:
        print(total_vec, vector_size)
        fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            if binary:
                row = row.astype(REAL)
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))